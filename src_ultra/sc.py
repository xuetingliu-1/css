
import os
import re
import json
import csv
import time
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]


def read_jsonl_keep_order(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                rows.append({})
            else:
                rows.append(json.loads(line))
    return rows


def load_medqa_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Your MedQA jsonl fields:
      - question / stem
      - options / choices : dict(A-E) or list
      - answer_idx : 'A'..'E' or 'A'..'D'
    """
    rows = read_jsonl_keep_order(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not r:
            out.append({"question": "", "options": {}, "answer_idx": "Z"})
            continue

        q = (r.get("question") or r.get("stem") or "").strip()
        raw_opts = r.get("options") or r.get("choices") or {}

        opts_dict: Dict[str, str] = {}
        if isinstance(raw_opts, list):
            letters = LETTER_5 if len(raw_opts) == 5 else LETTER_4
            for i, v in enumerate(raw_opts):
                if i < len(letters):
                    opts_dict[letters[i]] = str(v)
        elif isinstance(raw_opts, dict):
            for k in LETTER_5:
                if k in raw_opts and raw_opts[k] is not None:
                    vv = str(raw_opts[k]).strip()
                    if vv != "":
                        opts_dict[k] = vv

        gt = str(r.get("answer_idx", "Z")).strip().upper()
        out.append({"question": q, "options": opts_dict, "answer_idx": gt})
    return out


def load_medbullets_csv(path: str) -> List[Dict[str, Any]]:
    """
    MedBullets CSV columns:
      question, opa..op(e), answer_idx
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question", "") or "").strip()
            opts: Dict[str, str] = {}
            if (row.get("opa", "") or "").strip() != "": opts["A"] = row.get("opa", "").strip()
            if (row.get("opb", "") or "").strip() != "": opts["B"] = row.get("opb", "").strip()
            if (row.get("opc", "") or "").strip() != "": opts["C"] = row.get("opc", "").strip()
            if (row.get("opd", "") or "").strip() != "": opts["D"] = row.get("opd", "").strip()
            if (row.get("ope", "") or "").strip() != "": opts["E"] = row.get("ope", "").strip()
            gt = str(row.get("answer_idx", "Z")).strip().upper()
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out



def format_mcq(question: str, options: Dict[str, str]) -> str:
    q = (question or "").strip()
    lines = [q] if q else []
    for k in LETTER_5:
        if k in options and str(options.get(k, "")).strip() != "":
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines).strip()


def valid_letters_from_options(options: Dict[str, str]) -> List[str]:
    if "E" in options and str(options.get("E", "")).strip() != "":
        return LETTER_5
    return LETTER_4


def build_user_content(question: str,
                       options: Dict[str, str],
                       mode: str,
                       answer_tag: bool,
                       letter_only: bool) -> str:
    base = format_mcq(question, options)

    if mode == "cot":
        base += "\n\nLet's think step by step."

    if answer_tag:
        base += "\n\nAnswer:"

    if letter_only:
        base += "\n(Reply with only one letter: A/B/C/D/E.)"

    return base.strip()


_RE_ANSWER_LINE = re.compile(r"(?:^|\n)\s*Answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_SO_ANSWER = re.compile(r"\b(?:so\s*,?\s*)?(?:the\s*)?answer\s*(?:is|:)\s*([A-E])\b", re.IGNORECASE)
_RE_FINAL = re.compile(r"\bfinal\s*answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_PAREN = re.compile(r"\(([A-E])\)")
_RE_SINGLE = re.compile(r"\b([A-E])\b")

def extract_letter(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    t = text.strip()

    for pat in (_RE_ANSWER_LINE, _RE_SO_ANSWER, _RE_FINAL):
        m = pat.search(t)
        if m:
            c = m.group(1).upper()
            if c in valid_letters:
                return c

    m = _RE_PAREN.search(t)
    if m:
        c = m.group(1).upper()
        if c in valid_letters:
            return c

    hits = _RE_SINGLE.findall(t.upper())
    for h in reversed(hits):
        if h in valid_letters:
            return h

    return "Z"


def majority_vote(letters: List[str], valid_letters: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Majority vote; tie-break alphabetical (A < B < ...).
    Return: (winner, counts_dict)
    """
    filt = [x for x in letters if x in valid_letters]
    cnt = Counter(filt)
    if not cnt:
        return "Z", {}
  
    winner = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return winner, dict(cnt)


def pick_representative_text(cands_text: List[str], cands_letter: List[str], winner: str) -> str:
    """
    Choose the first candidate whose extracted letter == winner.
    Fallback to first candidate.
    """
    for t, l in zip(cands_text, cands_letter):
        if l == winner:
            return t
    return cands_text[0] if cands_text else ""

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True,
                    help="HF id or local path, e.g. /home/x/.../Llama-3-8B-UltraMedical")
    ap.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    ap.add_argument("--data_path", type=str, required=True)

    ap.add_argument("--mode", type=str, choices=["sp", "cot"], required=True)

  
    ap.add_argument("--decode", type=str, choices=["greedy", "sc"], default="greedy")
    ap.add_argument("--num_samples", type=int, default=10, help="K for SC (only used when --decode sc)")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

   
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["auto", "half", "bfloat16", "float16", "float32"])

  
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

 
    ap.add_argument("--answer_tag", action="store_true")
    ap.add_argument("--letter_only", action="store_true")

   
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--debug_first_n", type=int, default=0)

    
    ap.add_argument("--drop_option_e", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    
    if args.data_type == "medqa_jsonl":
        data = load_medqa_jsonl(args.data_path)
    else:
        data = load_medbullets_csv(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)


    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,  
    )

    stop = ["<|eot_id|>"]

   
    if args.decode == "greedy":
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            stop=stop,
            seed=args.seed,
            n=1,
        )
    else:
      
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            stop=stop,
            seed=args.seed,
            n=args.num_samples,
        )

    # outputs
    res_path = os.path.join(args.out_dir, f"{args.split_name}_res.txt")
    pred_path = os.path.join(args.out_dir, f"{args.split_name}_pred.txt")
    detail_path = os.path.join(args.out_dir, f"{args.split_name}_detail.jsonl")

    total = len(data)
    processed = 0
    t0 = time.time()
    print(f"Total examples: {total}", flush=True)

    with open(res_path, "w", encoding="utf-8") as f_res, \
         open(pred_path, "w", encoding="utf-8") as f_pred, \
         open(detail_path, "w", encoding="utf-8") as f_det:

        for batch in batch_iter(list(enumerate(data)), args.batch_size):
            convs: List[List[Dict[str, str]]] = []
            metas: List[Tuple[int, Dict[str, Any]]] = []

          
            for idx, ex in batch:
                q = ex.get("question", "")
                opts = dict(ex.get("options", {}) or {})
                if args.drop_option_e and "E" in opts:
                    opts.pop("E", None)

                user_content = build_user_content(
                    question=q,
                    options=opts,
                    mode=args.mode,
                    answer_tag=args.answer_tag,
                    letter_only=args.letter_only,
                )
                convs.append([{"role": "user", "content": user_content}])
                metas.append((idx, {**ex, "options": opts}))

            prompts = tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                add_generation_prompt=True
            )
            if isinstance(prompts, str):
                prompts = [prompts]

            outputs = llm.generate(prompts=prompts, sampling_params=sampling)

            
            for i, out in enumerate(outputs):
                idx, ex = metas[i]
                opts = ex.get("options", {}) or {}
                valid_letters = valid_letters_from_options(opts)

                cand_texts: List[str] = []
                cand_letters: List[str] = []

                if out.outputs:
                    for cand in out.outputs:
                        txt = (cand.text or "").strip()
                        cand_texts.append(txt)
                        cand_letters.append(extract_letter(txt, valid_letters))
                else:
                    cand_texts = [""]
                    cand_letters = ["Z"]

                if args.decode == "greedy":
                    final_pred = cand_letters[0]
                    vote_counts = {}
                    chosen_text = cand_texts[0]
                else:
                    final_pred, vote_counts = majority_vote(cand_letters, valid_letters)
                    chosen_text = pick_representative_text(cand_texts, cand_letters, final_pred)

                if args.debug_first_n > 0 and idx < args.debug_first_n:
                    print(f"\n--- DEBUG idx={idx} ---")
                    print("USER:\n", convs[i][0]["content"])
                    print("CAND LETTERS:", cand_letters)
                    print("VOTES:", vote_counts)
                    print("FINAL:", final_pred, "GT:", ex.get("answer_idx", "Z"))
                    print("CHOSEN (head):\n", chosen_text[:800])
                    print("---------------\n", flush=True)

                f_res.write(chosen_text.replace("\n", "\\n") + "\n")
                f_pred.write(final_pred + "\n")
                f_det.write(json.dumps({
                    "idx": idx,
                    "mode": args.mode,
                    "decode": args.decode,
                    "pred": final_pred,
                    "gt": ex.get("answer_idx", "Z"),
                    "question": ex.get("question", ""),
                    "options": opts,
                    "valid_letters": valid_letters,
                    "chosen_generation": chosen_text,
                    "cand_preds": cand_letters,
                    "vote_counts": vote_counts,
                    "generations": cand_texts,
                    "sampling": {
                        "max_tokens": args.max_tokens,
                        "stop": stop,
                        "temperature": 0.0 if args.decode == "greedy" else args.temperature,
                        "top_p": 1.0 if args.decode == "greedy" else args.top_p,
                        "top_k": -1 if args.decode == "greedy" else args.top_k,
                        "seed": args.seed,
                        "n": 1 if args.decode == "greedy" else args.num_samples,
                    },
                    "answer_tag": bool(args.answer_tag),
                    "letter_only": bool(args.letter_only),
                    "drop_option_e": bool(args.drop_option_e),
                }, ensure_ascii=False) + "\n")

            processed += len(metas)
            if processed % args.log_every == 0 or processed == total:
                dt = time.time() - t0
                speed = processed / dt if dt > 0 else 0.0
                print(f"[{processed}/{total}] {speed:.2f} ex/s | elapsed {dt/60:.1f} min", flush=True)

    print("Saved:")
    print(" ", res_path)
    print(" ", pred_path)
    print(" ", detail_path)


if __name__ == "__main__":
    main()

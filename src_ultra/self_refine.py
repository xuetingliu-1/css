
import os
import re
import json
import csv
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]

def read_jsonl_keep_order(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                rows.append({})
            else:
                rows.append(json.loads(line))
    return rows

def load_medqa_jsonl(path: str) -> List[Dict[str, Any]]:
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
                if i < len(letters) and v is not None and str(v).strip() != "":
                    opts_dict[letters[i]] = str(v).strip()
        elif isinstance(raw_opts, dict):
            for k in LETTER_5:
                if k in raw_opts and raw_opts[k] is not None and str(raw_opts[k]).strip() != "":
                    opts_dict[k] = str(raw_opts[k]).strip()
        gt = str(r.get("answer_idx", "Z")).strip().upper()
        out.append({"question": q, "options": opts_dict, "answer_idx": gt})
    return out

def load_medbullets_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            opts: Dict[str, str] = {}
            for col, letter in [("opa", "A"), ("opb", "B"), ("opc", "C"), ("opd", "D"), ("ope", "E")]:
                v = row.get(col, "")
                if v is None:
                    v = ""
                v = str(v).strip()
                if v != "":
                    opts[letter] = v
            gt = str(row.get("answer_idx", "Z")).strip().upper()
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out

def format_mcq_ultra(question: str, options: Dict[str, str]) -> str:
    lines: List[str] = []
    q = (question or "").strip()
    if q:
        lines.append(q)
    for k in LETTER_5:
        if k in options and options[k] is not None and str(options[k]).strip() != "":
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines).strip()

def valid_letters_from_options(options: Dict[str, str]) -> List[str]:
    if "E" in options and str(options.get("E", "")).strip() != "":
        return LETTER_5
    return LETTER_4


_RE_ANSWER_LINE = re.compile(r"(?:^|\n)\s*Answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_FINAL = re.compile(r"\bfinal\s*answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_SINGLE = re.compile(r"\b([A-E])\b")
_RE_STOP = re.compile(r"Stop:\s*(YES|NO)", re.IGNORECASE)

def extract_choice(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    t = text.strip()

    for pat in (_RE_ANSWER_LINE, _RE_FINAL):
        m = pat.search(t)
        if m:
            c = m.group(1).upper()
            if c in valid_letters:
                return c

    hits = _RE_SINGLE.findall(t.upper())
    for h in reversed(hits):
        if h in valid_letters:
            return h
    return "Z"

def parse_stop_flag(feedback: str) -> bool:
    m = _RE_STOP.search(feedback or "")
    return (m is not None) and (m.group(1).upper() == "YES")


def sr_init_prompt(input_text: str, valid_letters: List[str]) -> str:
    letters = "/".join(valid_letters)
    return (
        "Given the following multiple-choice medical question, select the correct option.\n"
        f"Reply with ONLY one letter: {letters}.\n\n"
        f"{input_text}\n\n"
        "Answer:"
    ).strip()

def sr_feedback_prompt(input_text: str, proposed: str, valid_letters: List[str]) -> str:
    letters = "/".join(valid_letters)
    return (
        "You are an expert clinical reviewer.\n"
        "Given the question/options and the proposed answer, write actionable and specific feedback.\n"
        "If the proposed answer is clearly correct, write 'Stop: YES'. Otherwise write 'Stop: NO'.\n"
        "Output format:\n"
        "Feedback: <your feedback>\n"
        "Stop: YES/NO\n\n"
        f"{input_text}\n\n"
        f"Valid options: {letters}\n"
        f"Proposed answer: {proposed}\n\n"
        "Feedback:"
    ).strip()

def sr_refine_prompt(input_text: str, proposed: str, feedback: str, valid_letters: List[str]) -> str:
    letters = "/".join(valid_letters)
    return (
        "Revise the answer using the feedback.\n"
        f"In your final output, ONLY provide the letter of the best option ({letters}) with no extra text.\n\n"
        f"{input_text}\n\n"
        f"Proposed answer: {proposed}\n\n"
        f"{feedback}\n\n"
        "Final answer:"
    ).strip()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def apply_chat_template_batch(tokenizer, user_texts: List[str]) -> List[str]:
    convs = [[{"role": "user", "content": ut}] for ut in user_texts]
    prompts = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    if isinstance(prompts, str):
        return [prompts]
    return prompts

def vllm_generate_first(llm: LLM, prompts: List[str], sampling: SamplingParams) -> List[str]:
    outs = llm.generate(prompts=prompts, sampling_params=sampling)
    texts: List[str] = []
    for o in outs:
        if o.outputs:
            texts.append((o.outputs[0].text or "").strip())
        else:
            texts.append("")
    return texts

def make_sampling(args, stage_seed: int, max_tokens: int) -> SamplingParams:
    if args.do_sample:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            seed=stage_seed,
            n=1,
        )
    else:
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            seed=stage_seed,
            n=1,
        )

def self_refine_one(
    llm: LLM,
    tokenizer,
    input_text: str,
    valid_letters: List[str],
    args,
    example_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    trace: Dict[str, Any] = {"init": None, "iters": []}

    # INIT
    init_user = sr_init_prompt(input_text, valid_letters)
    init_prompt = apply_chat_template_batch(tokenizer, [init_user])[0]
    init_sampling = make_sampling(args, stage_seed=args.seed + example_idx * 10000 + 1, max_tokens=args.init_max_tokens)
    y = vllm_generate_first(llm, [init_prompt], init_sampling)[0].strip()
    trace["init"] = y

    # ITERATE
    for t in range(args.max_iters):
        fb_user = sr_feedback_prompt(input_text, y, valid_letters)
        fb_prompt = apply_chat_template_batch(tokenizer, [fb_user])[0]
        fb_sampling = make_sampling(args, stage_seed=args.seed + example_idx * 10000 + 100 + t, max_tokens=args.fb_max_tokens)
        fb = vllm_generate_first(llm, [fb_prompt], fb_sampling)[0].strip()

        stop = parse_stop_flag(fb)

        ref_user = sr_refine_prompt(input_text, y, fb, valid_letters)
        ref_prompt = apply_chat_template_batch(tokenizer, [ref_user])[0]
        ref_sampling = make_sampling(args, stage_seed=args.seed + example_idx * 10000 + 200 + t, max_tokens=args.refine_max_tokens)
        y_new = vllm_generate_first(llm, [ref_prompt], ref_sampling)[0].strip()

        trace["iters"].append({"t": t, "feedback": fb, "stop": stop, "refined": y_new})
        y = y_new

        if args.use_stop_flag and stop:
            break

    # final: map to letter
    final_letter = extract_choice(y, valid_letters)
    return final_letter, trace


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    p.add_argument("--data_path", type=str, required=True, help="a file or a directory")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="test")

    # self-refine loop
    p.add_argument("--max_iters", type=int, default=4)
    p.add_argument("--use_stop_flag", action="store_true")

    # decoding
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    # stage token budgets
    p.add_argument("--init_max_tokens", type=int, default=64)
    p.add_argument("--fb_max_tokens", type=int, default=192)
    p.add_argument("--refine_max_tokens", type=int, default=32)

    # vLLM engine
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "half", "bfloat16", "float16", "float32"])

    # batching/log
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--drop_option_e", action="store_true", help="force 4-option by dropping E if exists")

    p.add_argument("--save_traces", action="store_true")
    return p.parse_args()

def list_files(data_path: str, data_type: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    exts = [".jsonl"] if data_type == "medqa_jsonl" else [".csv"]
    files = []
    for fn in os.listdir(data_path):
        if any(fn.lower().endswith(e) for e in exts):
            files.append(os.path.join(data_path, fn))
    files.sort()
    return files

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    files = list_files(args.data_path, args.data_type)

    for fp in files:
        base = os.path.basename(fp)
        split = os.path.splitext(base)[0] if args.split_name == "test" else args.split_name

        if args.data_type == "medqa_jsonl":
            data = load_medqa_jsonl(fp)
        else:
            data = load_medbullets_csv(fp)

        pred_path = os.path.join(args.out_dir, f"{split}_pred.txt")
        detail_path = os.path.join(args.out_dir, f"{split}_detail.jsonl")

        total = len(data)
        processed = 0
        t0 = time.time()
        print(f"\n=== File: {fp} | total={total} ===", flush=True)

        with open(pred_path, "w", encoding="utf-8") as f_pred, \
             open(detail_path, "w", encoding="utf-8") as f_det:

            for batch in batch_iter(list(enumerate(data)), args.batch_size):
                for idx, ex in batch:
                    q = ex.get("question", "")
                    opts = dict(ex.get("options", {}) or {})
                    if args.drop_option_e and "E" in opts:
                        opts.pop("E", None)

                    input_text = format_mcq_ultra(q, opts)
                    valid_letters = valid_letters_from_options(opts)

                    final_letter, trace = self_refine_one(
                        llm=llm,
                        tokenizer=tokenizer,
                        input_text=input_text,
                        valid_letters=valid_letters,
                        args=args,
                        example_idx=idx,
                    )

                    f_pred.write(final_letter + "\n")

                    rec = {
                        "idx": idx,
                        "final_letter": final_letter,
                        "gt": ex.get("answer_idx", "Z"),
                        "valid_letters": valid_letters,
                        "question": q,
                        "options": opts,
                        "trace": trace if args.save_traces else None,
                        "config": {
                            "max_iters": args.max_iters,
                            "use_stop_flag": bool(args.use_stop_flag),
                            "do_sample": bool(args.do_sample),
                            "temperature": (args.temperature if args.do_sample else 0.0),
                            "top_k": (args.top_k if args.do_sample else -1),
                            "top_p": (args.top_p if args.do_sample else 1.0),
                            "seed": args.seed,
                            "init_max_tokens": args.init_max_tokens,
                            "fb_max_tokens": args.fb_max_tokens,
                            "refine_max_tokens": args.refine_max_tokens,
                        },
                    }
                    f_det.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    processed += 1
                    if processed % args.log_every == 0 or processed == total:
                        dt = time.time() - t0
                        speed = processed / dt if dt > 0 else 0.0
                        print(f"[{processed}/{total}] {speed:.2f} ex/s | elapsed {dt/60:.1f} min", flush=True)

        print("Saved:")
        print(" ", pred_path)
        print(" ", detail_path)

if __name__ == "__main__":
    main()

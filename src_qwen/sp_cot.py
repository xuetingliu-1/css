

import os
import re
import json
import csv
import time
import argparse
from typing import Dict, List, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]



def read_jsonl(path: str) -> List[dict]:
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
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not r:
            out.append({"question": "", "options": {}, "answer_idx": "Z"})
            continue

        q = (r.get("question") or r.get("stem") or "").strip()
        raw_opts = r.get("options") or r.get("choices") or {}

        opts_dict: Dict[str, str] = {}
        if isinstance(raw_opts, list):
            letters = LETTER_5 if len(raw_opts) >= 5 else LETTER_4
            for i, v in enumerate(raw_opts):
                if i < len(letters):
                    opts_dict[letters[i]] = str(v).strip()
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
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question", "") or "").strip()
            opts: Dict[str, str] = {}
            if row.get("opa", "") != "": opts["A"] = row.get("opa", "").strip()
            if row.get("opb", "") != "": opts["B"] = row.get("opb", "").strip()
            if row.get("opc", "") != "": opts["C"] = row.get("opc", "").strip()
            if row.get("opd", "") != "": opts["D"] = row.get("opd", "").strip()
            if row.get("ope", "") != "": opts["E"] = row.get("ope", "").strip()
            gt = str(row.get("answer_idx", "Z")).strip().upper()
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out



def format_options_block(options: Dict[str, str]) -> str:
    lines = []
    for k in LETTER_5:
        if k in options and str(options[k]).strip() != "":
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines).strip()


def get_valid_letters(options: Dict[str, str]) -> List[str]:
    
    if "E" in options and str(options.get("E", "")).strip() != "":
        return LETTER_5
    return LETTER_4


def build_messages(question: str, options: Dict[str, str], mode: str, use_system: bool) -> List[Dict[str, str]]:
    opt_block = format_options_block(options)

  
    system_msg = (
        "You are an expert medical reasoning assistant for USMLE-style multiple-choice questions."
    )

    if mode == "sp":
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Select the single best answer.\n"
            "Reply with ONLY one letter (A/B/C/D/E)."
        )
    elif mode == "cot":
        
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Think step by step internally, but DO NOT show your reasoning.\n"
            "Reply with ONLY one letter (A/B/C/D/E)."
        )
    else:
        raise ValueError(f"Unknown mode={mode}")

    if use_system:
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user}]
    return [{"role": "user", "content": user}]



_RE_ONLY_LETTER = re.compile(r"^\s*([A-E])\s*$")
_RE_ANY_LETTER = re.compile(r"\b([A-E])\b")

def extract_letter(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    t = text.strip()

    m = _RE_ONLY_LETTER.match(t)
    if m:
        c = m.group(1).upper()
        return c if c in valid_letters else "Z"

    hits = _RE_ANY_LETTER.findall(t.upper())
    for h in reversed(hits):
        if h in valid_letters:
            return h
    return "Z"



def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def slice_to_max_len(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int):
    # keep the rightmost tokens
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    return input_ids[:, -max_len:], attention_mask[:, -max_len:]


def get_stop_ids(tokenizer: AutoTokenizer) -> List[int]:
 
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))

    for tok in ["<|im_end|>", "<|eot_id|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(int(tid))
        except Exception:
            pass

   
    ids = list(dict.fromkeys(ids))
    return ids


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="local path to Qwen2.5-7B-Instruct")
    ap.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["sp", "cot"], required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_input_tokens", type=int, default=4096, help="truncate long prompts (right side kept)")

    ap.add_argument("--use_system", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)

    # decoding
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    # dtype / quant
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # load data
    if args.data_type == "medqa_jsonl":
        data = load_medqa_jsonl(args.data_path)
    else:
        data = load_medbullets_csv(args.data_path)

   
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if args.load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available, cannot use --load_in_4bit")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=(dtype or torch.float16))
        model_kwargs["quantization_config"] = bnb

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    eos_ids = get_stop_ids(tokenizer)

    res_path = os.path.join(args.out_dir, f"{args.split_name}_res.txt")
    pred_path = os.path.join(args.out_dir, f"{args.split_name}_pred.txt")
    detail_path = os.path.join(args.out_dir, f"{args.split_name}_detail.jsonl")

    def batch_iter(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    total = len(data)
    processed = 0
    t0 = time.time()
    print(f"Total examples: {total}", flush=True)

    with open(res_path, "w", encoding="utf-8") as f_res, \
         open(pred_path, "w", encoding="utf-8") as f_pred, \
         open(detail_path, "w", encoding="utf-8") as f_det:

        for batch in batch_iter(list(enumerate(data)), args.batch_size):
            messages_list = []
            metas: List[Tuple[int, Dict[str, Any]]] = []

            for idx, ex in batch:
                q = ex.get("question", "")
                opts = ex.get("options", {}) or {}
                msgs = build_messages(q, opts, args.mode, use_system=args.use_system)
                messages_list.append(msgs)
                metas.append((idx, ex))

            enc = tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            # truncate very long prompts
            input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, args.max_input_tokens)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=bool(args.do_sample),
            )
            if args.do_sample:
                gen_kwargs.update(dict(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                ))

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            for i, (idx, ex) in enumerate(metas):
                prompt_len = int(attention_mask[i].sum().item())
                gen_ids = out[i][prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                opts = ex.get("options", {}) or {}
                valid_letters = get_valid_letters(opts)
                pred = extract_letter(text, valid_letters)

                f_res.write(text.replace("\n", "\\n") + "\n")
                f_pred.write(pred + "\n")
                f_det.write(json.dumps({
                    "idx": idx,
                    "mode": args.mode,
                    "pred": pred,
                    "gt": ex.get("answer_idx", "Z"),
                    "question": ex.get("question", ""),
                    "options": opts,
                    "generation": text,
                    "eos_ids": eos_ids,
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

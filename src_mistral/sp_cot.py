

import os
import re
import json
import csv
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

import torch
from transformers import BitsAndBytesConfig  


try:
    from transformers import FineGrainedFP8Config
    _HAS_FP8CFG = True
except Exception:
    _HAS_FP8CFG = False


try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    _HAS_MISTRAL3 = True
except Exception:
    _HAS_MISTRAL3 = False


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
    out = []
    for r in rows:
        if not r:
            out.append({"question": "", "options": {}, "answer_idx": "Z"})
            continue
        q = r.get("question") or r.get("stem") or ""
        opts = r.get("options") or r.get("choices") or {}

        if isinstance(opts, list):
            letters = LETTER_5 if len(opts) == 5 else LETTER_4
            opts_dict = {letters[i]: str(opts[i]) for i in range(len(opts))}
        elif isinstance(opts, dict):
            keys = [k for k in LETTER_5 if k in opts]
            opts_dict = {k: str(opts[k]) for k in keys}
        else:
            opts_dict = {}

        gt = r.get("answer_idx") or "Z"
        out.append({"question": q, "options": opts_dict, "answer_idx": gt})
    return out

def load_medbullets_csv(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "")
            opts = {}
            if row.get("opa", "") != "": opts["A"] = row.get("opa", "")
            if row.get("opb", "") != "": opts["B"] = row.get("opb", "")
            if row.get("opc", "") != "": opts["C"] = row.get("opc", "")
            if row.get("opd", "") != "": opts["D"] = row.get("opd", "")
            if row.get("ope", "") != "": opts["E"] = row.get("ope", "")
            gt = row.get("answer_idx", "Z")
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out


def format_options_block(options: Dict[str, str]) -> str:
    lines = []
    for k in LETTER_5:
        if k in options:
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines)

def get_valid_letters(options: Dict[str, str]) -> List[str]:
    return LETTER_5 if "E" in options else LETTER_4

def build_messages(question: str, options: Dict[str, str], mode: str, use_system: bool) -> List[Dict[str, str]]:
    opt_block = format_options_block(options)

    system_msg = (
        "You are an expert in clinical reasoning for USMLE-style multiple-choice questions.\n\n"
    )

    if mode == "sp":
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Given a question and a list of options, select the correct answer from the options directly."
        )
    elif mode == "cot":
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Given a question and a list of options,"
            "Let’s think step by step to answer the question. "
            "Select the correct answer from the options directly."
        )
    else:
        raise ValueError(f"Unknown mode={mode}")

    if use_system:
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user}]
    else:
        return [{"role": "user", "content": user}]


_RE_FINAL = re.compile(r"(?:final\s*answer|answer)\s*[:\-]?\s*([A-E])", re.IGNORECASE)
_RE_SINGLE = re.compile(r"\b([A-E])\b")

def extract_letter(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    m = _RE_FINAL.search(text)
    if m:
        c = m.group(1).upper()
        if c in valid_letters:
            return c
    hits = _RE_SINGLE.findall(text.upper())
    for h in hits[::-1]:
        if h in valid_letters:
            return h
    return "Z"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def truncate_right(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int):
    
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    return input_ids[:, -max_len:], attention_mask[:, -max_len:]

def _require_mistral3():
    if not _HAS_MISTRAL3:
        raise RuntimeError(
            "Missing Mistral3ForConditionalGeneration / MistralCommonBackend.\n"
            "You likely need Transformers 'main' + mistral-common>=1.8.6, per the model card.\n"
            "Install:\n"
            "  pip install git+https://github.com/huggingface/transformers\n"
            "  pip install mistral-common --upgrade\n"
        )

def _load_tokenizer(model_path: str):
    _require_mistral3()
    tok = MistralCommonBackend.from_pretrained(model_path)
    # ensure right padding so our batching stays simple
    if hasattr(tok, "padding_side"):
        tok.padding_side = "right"
    # pad token safety (often eos-as-pad)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok

def _load_model(
    model_path: str,
    dtype: Optional[torch.dtype],
    device_map: str,
    load_in_4bit: bool,
    dequantize_fp8_to_bf16: bool,
):
    _require_mistral3()

    model_kwargs = dict(device_map=device_map)

    # dtype: prefer new 'dtype' kwarg if supported, fallback to torch_dtype
    if dtype is not None:
        try:
            model_kwargs["dtype"] = dtype  # newer transformers
            _ = Mistral3ForConditionalGeneration.from_pretrained  # just to silence linters
        except Exception:
            model_kwargs["torch_dtype"] = dtype  # older transformers

    # quantization
    if load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available, cannot use --load_in_4bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(dtype or torch.float16),
        )

    # FP8 -> BF16 dequant (HF card suggests FineGrainedFP8Config(dequantize=True) for BF16) 
    if dequantize_fp8_to_bf16:
        if not _HAS_FP8CFG:
            raise RuntimeError("FineGrainedFP8Config not available in your transformers; cannot use --dequantize_fp8_to_bf16")
        model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)

    model = Mistral3ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model

def _encode_batch(tokenizer, messages_list: List[List[Dict[str, str]]]):
 
    try:
        enc = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        if isinstance(enc, dict) and "input_ids" in enc:
            return enc
    except Exception:
        pass

  
    encoded = []
    for msgs in messages_list:
        e = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
       
        encoded.append(e)

    input_ids_list = [e["input_ids"][0] for e in encoded]
    attn_list = [e.get("attention_mask", torch.ones_like(e["input_ids"]))[0] for e in encoded]

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="local path to Ministral-3-*-Instruct-2512")
    ap.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["sp", "cot"], required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_input_tokens", type=int, default=4096, help="truncate long prompts (right side kept)")

    ap.add_argument("--use_system", action="store_true", help="include a system message")
    ap.add_argument("--log_every", type=int, default=20)

  
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")

   
    ap.add_argument("--dequantize_fp8_to_bf16", action="store_true",
                    help="Load FP8 weights then dequantize to BF16 using FineGrainedFP8Config(dequantize=True).")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

 
    if args.data_type == "medqa_jsonl":
        data = load_medqa_jsonl(args.data_path)
    else:
        data = load_medbullets_csv(args.data_path)

  
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

   
    tokenizer = _load_tokenizer(args.model_path)
    model = _load_model(
        args.model_path,
        dtype=dtype,
        device_map="auto",
        load_in_4bit=bool(args.load_in_4bit),
        dequantize_fp8_to_bf16=bool(args.dequantize_fp8_to_bf16),
    )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id

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

            enc = _encode_batch(tokenizer, messages_list)
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

            
            input_ids, attention_mask = truncate_right(input_ids, attention_mask, args.max_input_tokens)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
            )
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)

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

           
            input_len = input_ids.shape[1]
            for i, (idx, ex) in enumerate(metas):
                gen_ids = out[i][input_len:]
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

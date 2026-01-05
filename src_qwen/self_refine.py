

import os
import json
import csv
import re
import argparse
from typing import Dict, Any, List, Sequence, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_global_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(fp: str) -> Sequence[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def read_medbullets_csv(fp: str) -> List[Dict[str, Any]]:
  
    rows: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
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
            rows.append({
                "question": q,
                "options": opts,
                "answer_idx": (row.get("answer_idx") or "").strip().upper() if row.get("answer_idx") is not None else ""
            })
    return rows


def read_medbullets_json_columnwise(fp: str) -> List[Dict[str, Any]]:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in MedBullets json, got {type(obj)}: {fp}")

    qcol = obj.get("question")
    if not isinstance(qcol, dict):
        raise ValueError(f"Expected obj['question'] to be dict-of-rows in {fp}")

    def _to_int(s: str) -> int:
        try:
            return int(s)
        except Exception:
            return 10**18

    idx_keys = list(qcol.keys())
    idx_keys.sort(key=_to_int)

    rows: List[Dict[str, Any]] = []
    for ik in idx_keys:
        q = str(qcol.get(ik, "")).strip()

        opts: Dict[str, str] = {}
        for col, letter in [("opa", "A"), ("opb", "B"), ("opc", "C"), ("opd", "D"), ("ope", "E")]:
            col_dict = obj.get(col, {})
            v = ""
            if isinstance(col_dict, dict):
                v = col_dict.get(ik, "")
            if v is None:
                v = ""
            v = str(v).strip()
            if v != "":
                opts[letter] = v

        ans_col = obj.get("answer_idx", {})
        ans = ""
        if isinstance(ans_col, dict):
            ans = str(ans_col.get(ik, "")).strip().upper()

        rows.append({"question": q, "options": opts, "answer_idx": ans})
    return rows


def load_dataset(data_type: str, path: str) -> List[Dict[str, Any]]:
    if data_type == "medqa_jsonl":
        raw = read_jsonl(path)
        out: List[Dict[str, Any]] = []
        for r in raw:
            q = r.get("question") or r.get("stem") or ""
            opts = r.get("options") or r.get("choices") or {}

            opts_dict: Dict[str, str] = {}
            if isinstance(opts, dict):
                for k in sorted(opts.keys()):
                    kk = str(k).strip().upper()
                    if kk in ["A", "B", "C", "D", "E"]:
                        vv = opts.get(k, "")
                        vv = "" if vv is None else str(vv).strip()
                        if vv != "":
                            opts_dict[kk] = vv
            elif isinstance(opts, list):
                for i, v in enumerate(opts[:5]):
                    kk = chr(ord("A") + i)
                    vv = "" if v is None else str(v).strip()
                    if vv != "":
                        opts_dict[kk] = vv

            out.append({
                "question": str(q),
                "options": opts_dict,
                "answer_idx": (r.get("answer_idx") or "").strip().upper() if r.get("answer_idx") is not None else ""
            })
        return out

    if data_type == "medbullets_csv":
        return read_medbullets_csv(path)

    if data_type == "medbullets_json":
        return read_medbullets_json_columnwise(path)

    raise ValueError(f"Unsupported data_type={data_type}")



def format_options_medins(raw_options) -> str:
    """
    MedIns style: "A: ... \t B: ... \t C: ..."
    """
    if isinstance(raw_options, dict):
        pieces = []
        for k in sorted(raw_options.keys()):
            pieces.append(f"{k}: {raw_options[k]}")
        return "\t".join(pieces)
    if isinstance(raw_options, list):
        pieces = []
        for i, content in enumerate(raw_options):
            k = chr(ord("A") + i)
            pieces.append(f"{k}: {content}")
        return "\t".join(pieces)
    return ""


def build_input_text(question: str, options_str: str) -> str:
    return f"Question: {question}\nOptions: {options_str}"



def extract_choice(text: str) -> str:
    if not text:
        return "Z"

 
    m = re.search(r"\b([A-E])\b", text.strip())
    if m:
        return m.group(1).upper()

    m = re.search(r"right answer is\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"answer\s*(is|:)?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()

    m = re.search(r"option\s*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return "Z"


def parse_stop_flag(feedback: str) -> bool:
    m = re.search(r"Stop:\s*(YES|NO)", feedback, re.IGNORECASE)
    return (m is not None) and (m.group(1).upper() == "YES")



def sr_init_instruction() -> str:
    return (
        "Given a question and a list of options, select the correct answer from the options directly."
    )


def sr_feedback_instruction() -> str:
    return (
        "You are an expert clinical reviewer.\n"
        "Given the question, options, and the proposed answer, write actionable and specific feedback.\n"
        "If the proposed answer is clearly correct, write 'Stop: YES'. Otherwise write 'Stop: NO'.\n"
        "Output format:\n"
        "Feedback: <your feedback>\n"
        "Stop: YES/NO"
    )


def sr_refine_instruction() -> str:
    return (
        "Revise the answer using the feedback.\n"
        "In your final output, ONLY provide the letter of the best option (A, B, C, D, or E) "
        "without any additional text."
    )


def get_terminators(tokenizer) -> List[int]:
    """
    Qwen2.5 typically uses eos, and may include <|im_end|>.
    """
    terminators: List[int] = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))
    for tok in ["<|im_end|>", "<|eot_id|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid not in terminators:
                terminators.append(int(tid))
        except Exception:
            pass
    return list(dict.fromkeys(terminators))


def slice_to_max_len(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], max_len: int):
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    input_ids = input_ids[:, -max_len:]
    if attention_mask is not None:
        attention_mask = attention_mask[:, -max_len:]
    return input_ids, attention_mask


def build_messages(system_inst: str, user_query: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_query},
    ]


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    terminators: List[int],
    pad_id: int,
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:

    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, max_input_tokens)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_k=top_k if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        eos_token_id=terminators,
        pad_token_id=pad_id,
        use_cache=True,
    )

    if attention_mask is not None:
        prompt_len = int(attention_mask[0].sum().item())
    else:
        prompt_len = input_ids.shape[1]

    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()



def self_refine_one(
    model,
    tokenizer,
    terminators: List[int],
    pad_id: int,
    input_text: str,
    args,
    example_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    trace: Dict[str, Any] = {"init": None, "iters": []}

    if args.seed is not None:
        set_global_seed(args.seed + example_idx * 10000)

    # INIT
    y = generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=build_messages(sr_init_instruction(), input_text),
        terminators=terminators,
        pad_id=pad_id,
        max_new_tokens=args.init_max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    ).strip()
    trace["init"] = y

    # ITERATE
    for t in range(args.max_iters):
        if args.seed is not None:
            set_global_seed(args.seed + example_idx * 10000 + t * 10 + 1)

        fb_query = f"{input_text}\n\nProposed answer:\n{y}\n\nFeedback:\n"
        fb = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=build_messages(sr_feedback_instruction(), fb_query),
            terminators=terminators,
            pad_id=pad_id,
            max_new_tokens=args.fb_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ).strip()

        stop = parse_stop_flag(fb)

        if args.seed is not None:
            set_global_seed(args.seed + example_idx * 10000 + t * 10 + 2)

        refine_query = f"{input_text}\n\nProposed answer:\n{y}\n\nFeedback:\n{fb}\n\nFinal answer:\n"
        y_new = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=build_messages(sr_refine_instruction(), refine_query),
            terminators=terminators,
            pad_id=pad_id,
            max_new_tokens=args.refine_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ).strip()

        trace["iters"].append({"t": t, "feedback": fb, "stop": stop, "refined": y_new})
        y = y_new

        if args.use_stop_flag and stop:
            break

    return y, trace


def run_file(model, tokenizer, terminators, pad_id, data_type: str, fp: str, save_dir: str, args):
    ensure_dir(save_dir)
    base = os.path.basename(fp)
    split = os.path.splitext(base)[0]

    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_res = os.path.join(save_dir, f"{split}_res.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = load_dataset(data_type, fp)
    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    with open(out_pred, "w", encoding="utf-8") as f_pred, open(out_res, "w", encoding="utf-8") as f_res:
        for idx, ex in enumerate(tqdm(data, desc=f"Qwen25 Self-Refine {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str = format_options_medins(raw_options)
            input_text = build_input_text(str(q), str(options_str))

            final_raw, trace = self_refine_one(
                model=model,
                tokenizer=tokenizer,
                terminators=terminators,
                pad_id=pad_id,
                input_text=input_text,
                args=args,
                example_idx=idx,
            )
            final_letter = extract_choice(final_raw)

            f_pred.write(final_letter + "\n")
            f_res.write(final_raw.replace("\n", "\\n") + "\n")

            if ftrace is not None:
                rec = {
                    "idx": idx,
                    "data_type": data_type,
                    "final_letter": final_letter,
                    "final_raw": final_raw,
                    "gt": ex.get("answer_idx", ""),
                    "question": ex.get("question", ""),
                    "options": raw_options,
                    "trace": trace,
                }
                ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if ftrace is not None:
        ftrace.close()

    print("Saved:")
    print(" ", out_pred)
    print(" ", out_res)
    if args.save_traces:
        print(" ", out_trace)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True,
                   help="local path or HF id, e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_input_tokens", type=int, default=4096,
                   help="truncate long prompts (keep rightmost tokens)")

    p.add_argument("--data_type", type=str, required=True,
                   choices=["medqa_jsonl", "medbullets_csv", "medbullets_json"])
    p.add_argument("--data_path", type=str, required=True,
                   help="a single file (.jsonl/.csv/.json) or a directory containing those files")
    p.add_argument("--save_dir", type=str, required=True)

   
    p.add_argument("--max_iters", type=int, default=4)
    p.add_argument("--use_stop_flag", action="store_true",
                   help="stop early when feedback contains 'Stop: YES'")

  
    p.add_argument("--do_sample", action="store_true",
                   help="turn on sampling for ALL stages (init/feedback/refine). default is greedy for all stages")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)


    p.add_argument("--init_max_new_tokens", type=int, default=64)
    p.add_argument("--fb_max_new_tokens", type=int, default=192)
    p.add_argument("--refine_max_new_tokens", type=int, default=32)


    p.add_argument("--seed", type=int, default=1234,
                   help="seed for reproducible sampling; also used to make stages deterministic per example")

    p.add_argument("--save_traces", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    terminators = get_terminators(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

   
    if os.path.isfile(args.data_path):
        files = [args.data_path]
    else:
        exts = (".jsonl", ".csv", ".json")
        files = [
            os.path.join(args.data_path, fn)
            for fn in os.listdir(args.data_path)
            if fn.lower().endswith(exts)
        ]
        files.sort()

    for fp in files:
        run_file(
            model=model,
            tokenizer=tokenizer,
            terminators=terminators,
            pad_id=pad_id,
            data_type=args.data_type,
            fp=fp,
            save_dir=args.save_dir,
            args=args,
        )


if __name__ == "__main__":
    main()

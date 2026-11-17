"""Shared method engine derived from the uploaded Ministral baseline."""
from cssbench.prompts.self_refine import format_options_medins, build_input_text, sr_init_instruction, sr_feedback_instruction, sr_refine_instruction, build_messages
from cssbench.backend import load_tokenizer, load_tokenizer as _load_tokenizer, load_model, load_model as _load_model, encode_batch as _encode_batch, load_css_model as load_ministral3


import os
import json
import csv
import re
import argparse
from typing import Dict, Any, List, Sequence, Tuple, Optional

import torch
from tqdm import tqdm










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
    """
    MedBullets CSV columns:
      question, opa, opb, opc, opd, (ope), answer_idx(optional)
    """
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















def slice_to_max_len(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], max_len: int):
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    input_ids = input_ids[:, -max_len:]
    if attention_mask is not None:
        attention_mask = attention_mask[:, -max_len:]
    return input_ids, attention_mask






@torch.inference_mode()
def generate_batch_text(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
):
    enc = _encode_batch(tokenizer, messages_list)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, max_input_tokens)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
    )
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.stop_token_ids

    if do_sample:
        gen_kwargs.update(dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ))

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs
    )

    texts: List[str] = []
    for i in range(out.shape[0]):
        prompt_len = input_ids.shape[1]  # generate() returns the full padded input prefix
        gen_ids = out[i][prompt_len:]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    return generate_batch_text(
        model=model,
        tokenizer=tokenizer,
        messages_list=[messages],
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )[0]

def self_refine_one(
    model,
    tokenizer,
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


def self_refine_batch_greedy(
    model,
    tokenizer,
    input_texts: List[str],
    args,
    example_indices: List[int],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    
    assert not args.do_sample, "Batch self-refine only supports greedy (do_sample=False)."

    traces: List[Dict[str, Any]] = [{"init": None, "iters": []} for _ in input_texts]

  
    init_msgs = [build_messages(sr_init_instruction(), it) for it in input_texts]
    ys = generate_batch_text(
        model=model,
        tokenizer=tokenizer,
        messages_list=init_msgs,
        max_new_tokens=args.init_max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        do_sample=False,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    ys = [y.strip() for y in ys]
    for k in range(len(ys)):
        traces[k]["init"] = ys[k]

    active = [True for _ in ys]

    for t in range(args.max_iters):
        active_idx = [i for i, a in enumerate(active) if a]
        if not active_idx:
            break

       
        fb_msgs = []
        for bi in active_idx:
            fb_query = f"{input_texts[bi]}\n\nProposed answer:\n{ys[bi]}\n\nFeedback:\n"
            fb_msgs.append(build_messages(sr_feedback_instruction(), fb_query))

        fbs = generate_batch_text(
            model=model,
            tokenizer=tokenizer,
            messages_list=fb_msgs,
            max_new_tokens=args.fb_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        fbs = [x.strip() for x in fbs]

        
        refine_msgs = []
        stops = []
        for j, bi in enumerate(active_idx):
            fb = fbs[j]
            stop = parse_stop_flag(fb)
            stops.append(stop)
            refine_query = f"{input_texts[bi]}\n\nProposed answer:\n{ys[bi]}\n\nFeedback:\n{fb}\n\nFinal answer:\n"
            refine_msgs.append(build_messages(sr_refine_instruction(), refine_query))

        ys_new = generate_batch_text(
            model=model,
            tokenizer=tokenizer,
            messages_list=refine_msgs,
            max_new_tokens=args.refine_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        ys_new = [x.strip() for x in ys_new]

        for j, bi in enumerate(active_idx):
            traces[bi]["iters"].append({"t": t, "feedback": fbs[j], "stop": stops[j], "refined": ys_new[j]})
            ys[bi] = ys_new[j]
            if args.use_stop_flag and stops[j]:
                active[bi] = False

    return ys, traces


def run_file(model, tokenizer, data_type: str, fp: str, save_dir: str, args):
    ensure_dir(save_dir)
    base = os.path.basename(fp)
    split = os.path.splitext(base)[0]

    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_res = os.path.join(save_dir, f"{split}_res.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = load_dataset(data_type, fp)
    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    # sampling + batch_size>1 is not reproducible per-example with transformers generate
    if args.do_sample and args.batch_size > 1:
        print("[WARN] --do_sample with --batch_size>1 is not supported for reproducible per-example seeding. "
              "Forcing batch_size=1.")
        args.batch_size = 1

    with open(out_pred, "w", encoding="utf-8") as f_pred, open(out_res, "w", encoding="utf-8") as f_res:
        if (not args.do_sample) and args.batch_size > 1:
            # greedy batched pipeline
            bs = int(args.batch_size)
            for st in tqdm(range(0, len(data), bs), desc=f"Self-Refine (greedy,batch={bs}) {base}"):
                batch = data[st: st + bs]
                input_texts = []
                ex_indices = list(range(st, st + len(batch)))

                for ex in batch:
                    q = ex.get("question") or ex.get("stem") or ""
                    raw_options = ex.get("options") or ex.get("choices") or {}
                    options_str = format_options_medins(raw_options)
                    input_texts.append(build_input_text(str(q), str(options_str)))

                finals_raw, traces = self_refine_batch_greedy(
                    model=model,
                    tokenizer=tokenizer,
                    input_texts=input_texts,
                    args=args,
                    example_indices=ex_indices,
                )

                for local_i, (final_raw, trace, ex) in enumerate(zip(finals_raw, traces, batch)):
                    idx = st + local_i
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
                            "options": ex.get("options", {}),
                            "trace": trace,
                        }
                        ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            
            for idx, ex in enumerate(tqdm(data, desc=f"Self-Refine {base}")):
                q = ex.get("question") or ex.get("stem") or ""
                raw_options = ex.get("options") or ex.get("choices") or {}
                options_str = format_options_medins(raw_options)
                input_text = build_input_text(str(q), str(options_str))

                final_raw, trace = self_refine_one(
                    model=model,
                    tokenizer=tokenizer,
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
                   help="local path or HF id, e.g. mistralai/Ministral-3-8B-Instruct-2512")
    p.add_argument("--data_type", type=str, required=True,
                   choices=["medqa_jsonl", "medbullets_csv", "medbullets_json"])
    p.add_argument("--data_path", type=str, required=True,
                   help="a single file (.jsonl/.csv/.json) or a directory containing those files")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--max_input_tokens", type=int, default=4096,
                   help="truncate long prompts (keep rightmost tokens)")

   
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

    
    p.add_argument("--batch_size", type=int, default=1,
                   help="batch across examples for greedy self-refine. "
                        "If --do_sample is set, this will be forced to 1.")


    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--dequantize_fp8_to_bf16", action="store_true",
                   help="Load FP8 weights then dequantize to BF16 using FineGrainedFP8Config(dequantize=True).")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="(optional) load in 4bit via bitsandbytes (may reduce quality).")

    # Logging
    p.add_argument("--save_traces", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    tokenizer = _load_tokenizer(args.model_path)
    model = _load_model(
        model_path=args.model_path,
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        dequantize_fp8_to_bf16=bool(args.dequantize_fp8_to_bf16),
    )

    # allow file or directory
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
            data_type=args.data_type,
            fp=fp,
            save_dir=args.save_dir,
            args=args,
        )


if __name__ == "__main__":
    main()

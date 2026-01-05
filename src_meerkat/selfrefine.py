
import os
import json
import csv
import argparse
import re
from typing import Dict, Any, List, Tuple, Sequence, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM



USMLE_SYSTEM_COT = (
    "The following is a multiple-choice question about medical knowledge. "
    "Solve this in a step-by-step fashion, starting by summarizing the available information. "
    "Output a single option from the given options as the final answer. "
    "You are strongly required to follow the specified output format; conclude your response "
    "with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"
)

USMLE_SYSTEM_SP = (
    "The following is a multiple-choice question about medical knowledge. "
    "Output a single option from the given options as the final answer. "
    "You are strongly required to follow the specified output format; conclude your response "
    "with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"
)

def sr_init_system(prompt_type: str) -> str:
  
    return USMLE_SYSTEM_COT if prompt_type == "cot" else USMLE_SYSTEM_SP

def sr_feedback_system() -> str:
    return (
        "You are an expert clinical reviewer.\n"
        "Given the question, options, and the proposed answer, write actionable and specific feedback.\n"
        "If the proposed answer is clearly correct, write 'Stop: YES'. Otherwise write 'Stop: NO'.\n"
        "Output format:\n"
        "Feedback: <your feedback>\n"
        "Stop: YES/NO\n"
    )

def sr_refine_system() -> str:
    return (
        "Revise the answer using the feedback.\n"
        "You MUST end with the exact phrase: \"the answer is ([option_id]) [answer_string]\".\n"
    )



def extract_answer_letter(text: str, options: str = "ABCDE") -> str:
    m = re.findall(rf"the answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(rf"answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.search(rf"\b([{options}])\b", text)
    return m.group(1).upper() if m else "Z"

def parse_stop_flag(feedback: str) -> bool:
    m = re.search(r"Stop:\s*(YES|NO)", feedback or "", flags=re.IGNORECASE)
    return (m is not None) and (m.group(1).upper() == "YES")



def format_options_parentheses(raw_options) -> Tuple[str, str]:
    
    items: List[Tuple[str, str]] = []

    if isinstance(raw_options, dict):
        for k in sorted(raw_options.keys()):
            kk = str(k).strip().upper()
            if kk in "ABCDE":
                vv = raw_options[k]
                if vv is None:
                    vv = ""
                vv = str(vv).strip()
                if vv != "":
                    items.append((kk, vv))
    elif isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            k = chr(ord("A") + i)
            if k in "ABCDE":
                if v is None:
                    v = ""
                vv = str(v).strip()
                if vv != "":
                    items.append((k, vv))

    option_letters = "".join([k for k, _ in items])
    options_str = " ".join([f"({k}) {v}" for k, v in items])
    return options_str, (option_letters if option_letters else "ABCD")

def read_jsonl(fp: str) -> List[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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
            rows.append({"question": q, "options": opts, "answer_idx": (row.get("answer_idx") or "").strip().upper()})
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

def read_dataset_any(fp: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(fp)
    if ext == ".csv":
        return read_medbullets_csv(fp)
    if ext == ".json":
        return read_medbullets_json_columnwise(fp)
    raise ValueError(f"Unsupported data file: {fp} (expect .jsonl/.csv/.json)")


def get_terminators(tokenizer) -> List[int]:
    terminators: List[int] = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            terminators.append(int(eot))
    except Exception:
        pass
    return list(dict.fromkeys(terminators))

def build_chat_input(tokenizer, system_msg: str, user_msg: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids

@torch.inference_mode()
def generate_once(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    terminators: List[int],
    pad_id: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_k=top_k if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        eos_token_id=terminators,
        pad_token_id=pad_id,
    )
    gen = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()



def self_reflect_one(
    model,
    tokenizer,
    terminators: List[int],
    pad_id: int,
    question: str,
    options_str: str,
    option_letters: str,
    prompt_type: str,
    args,
    ex_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    trace: Dict[str, Any] = {"init": None, "iters": []}

  
    base_user = (question + " " + options_str).strip()

  
    init_sys = sr_init_system(prompt_type)
    init_user = base_user
    input_ids = build_chat_input(tokenizer, init_sys, init_user).to(model.device)

    y = generate_once(
        model, tokenizer, input_ids, terminators, pad_id,
        max_new_tokens=args.init_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    trace["init"] = y


    for t in range(args.max_iters):
        # feedback stage
        fb_sys = sr_feedback_system()
        fb_user = (
            f"{base_user}\n\n"
            f"Proposed answer:\n{y}\n\n"
            "Feedback:\n"
        )
        fb_ids = build_chat_input(tokenizer, fb_sys, fb_user).to(model.device)

        fb = generate_once(
            model, tokenizer, fb_ids, terminators, pad_id,
            max_new_tokens=args.fb_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        stop = parse_stop_flag(fb)

        # refine stage
        ref_sys = sr_refine_system()
        ref_user = (
            f"{base_user}\n\n"
            f"Proposed answer:\n{y}\n\n"
            f"{fb}\n\n"
            "Revised solution:\n"
        )
        ref_ids = build_chat_input(tokenizer, ref_sys, ref_user).to(model.device)

        y_new = generate_once(
            model, tokenizer, ref_ids, terminators, pad_id,
            max_new_tokens=args.refine_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        trace["iters"].append({"t": t, "feedback": fb, "stop": stop, "refined": y_new})
        y = y_new

        if args.use_stop_flag and stop:
            break

    final_letter = extract_answer_letter(y, options=option_letters)
    return final_letter, trace



def infer_file(model, tokenizer, data_path: str, save_dir: str, prompt_type: str, args):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(data_path)
    split = os.path.splitext(base)[0]

    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_res = os.path.join(save_dir, f"{split}_res.txt")
    out_detail = os.path.join(save_dir, f"{split}_detail.jsonl")

    data = read_dataset_any(data_path)
    terminators = get_terminators(tokenizer)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with open(out_pred, "w", encoding="utf-8") as fpred, \
         open(out_res, "w", encoding="utf-8") as fres, \
         open(out_detail, "w", encoding="utf-8") as fdet:

        for i, ex in enumerate(tqdm(data, desc=f"Meerkat SELF-REFLECT {prompt_type.upper()} {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str, option_letters = format_options_parentheses(raw_options)

            if args.drop_option_e and "E" in option_letters:
               
                if isinstance(raw_options, dict):
                    raw2 = {k: v for k, v in raw_options.items() if str(k).upper() != "E"}
                elif isinstance(raw_options, list):
                    raw2 = raw_options[:4]
                else:
                    raw2 = raw_options
                options_str, option_letters = format_options_parentheses(raw2)

          
            if args.max_iters <= 0:
                sys_msg = sr_init_system(prompt_type)
                user_msg = (q + " " + options_str).strip()
                ids = build_chat_input(tokenizer, sys_msg, user_msg).to(model.device)
                text = generate_once(
                    model, tokenizer, ids, terminators, pad_id,
                    max_new_tokens=args.init_max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                letter = extract_answer_letter(text, options=option_letters)
                fpred.write(letter + "\n")
                fres.write(text.replace("\n", " ") + "\n")
                fdet.write(json.dumps({
                    "idx": i,
                    "final_letter": letter,
                    "final_raw": text,
                    "trace": {"init": text, "iters": []},
                    "gt": ex.get("answer_idx", "Z"),
                    "option_letters": option_letters,
                }, ensure_ascii=False) + "\n")
                continue

            # self-reflection loop
            final_letter, trace = self_reflect_one(
                model=model,
                tokenizer=tokenizer,
                terminators=terminators,
                pad_id=pad_id,
                question=q,
                options_str=options_str,
                option_letters=option_letters,
                prompt_type=prompt_type,
                args=args,
                ex_idx=i,
            )

          
            final_raw = trace["iters"][-1]["refined"] if trace["iters"] else trace["init"]

            fpred.write(final_letter + "\n")
            fres.write(final_raw.replace("\n", " ") + "\n")
            fdet.write(json.dumps({
                "idx": i,
                "final_letter": final_letter,
                "final_raw": final_raw,
                "trace": trace if args.save_traces else None,
                "gt": ex.get("answer_idx", "Z"),
                "option_letters": option_letters,
                "config": {
                    "prompt_type": prompt_type,
                    "max_iters": args.max_iters,
                    "use_stop_flag": bool(args.use_stop_flag),
                    "do_sample": bool(args.do_sample),
                    "temperature": args.temperature if args.do_sample else 0.0,
                    "top_k": args.top_k if args.do_sample else 0,
                    "top_p": args.top_p if args.do_sample else 1.0,
                    "init_max_new_tokens": args.init_max_new_tokens,
                    "fb_max_new_tokens": args.fb_max_new_tokens,
                    "refine_max_new_tokens": args.refine_max_new_tokens,
                    "drop_option_e": bool(args.drop_option_e),
                }
            }, ensure_ascii=False) + "\n")

    print("Saved pred  :", out_pred)
    print("Saved res   :", out_res)
    print("Saved detail:", out_detail)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="local path or HF id")
    p.add_argument("--data_path", type=str, required=True, help=".jsonl/.csv/.json file or a directory")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--prompt_type", type=str, default="sp", choices=["sp", "cot"])

    # self-reflection loop
    p.add_argument("--max_iters", type=int, default=4)
    p.add_argument("--use_stop_flag", action="store_true")
    p.add_argument("--save_traces", action="store_true")
    p.add_argument("--drop_option_e", action="store_true", help="force 4-option by dropping E if exists")

    # decoding for ALL stages
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    # token budgets (stage-wise)
    p.add_argument("--init_max_new_tokens", type=int, default=64)
    p.add_argument("--fb_max_new_tokens", type=int, default=192)
    p.add_argument("--refine_max_new_tokens", type=int, default=32)

    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()

def main():
    args = parse_args()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

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
        infer_file(
            model=model,
            tokenizer=tokenizer,
            data_path=fp,
            save_dir=args.save_dir,
            prompt_type=args.prompt_type,
            args=args,
        )

if __name__ == "__main__":
    main()

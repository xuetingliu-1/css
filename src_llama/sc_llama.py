
import os
import json
import csv
import re
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]


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


def set_seed(seed: int):
    
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_answer_letter(text: str, options: str = "ABCDE") -> str:
   
    if not text:
        return "Z"
    m = re.findall(rf"the answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.findall(rf"answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.search(rf"\b([{options}])\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "Z"


def majority_vote(letters: List[str], valid_letters: str) -> Tuple[str, Dict[str, int]]:
    
    cnt = Counter([x for x in letters if x in valid_letters])
    if not cnt:
        return "Z", {}
    winner = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return winner, dict(cnt)


def format_options_parentheses(raw_options) -> Tuple[str, str]:

    items: List[Tuple[str, str]] = []

    if isinstance(raw_options, dict):
        for k in sorted(raw_options.keys()):
            kk = str(k).strip().upper()
            if kk in "ABCDE":
                vv = raw_options.get(k, "")
                if vv is None:
                    vv = ""
                vv = str(vv).strip()
                if vv != "":
                    items.append((kk, vv))
    elif isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            kk = chr(ord("A") + i)
            if kk in "ABCDE":
                if v is None:
                    v = ""
                vv = str(v).strip()
                if vv != "":
                    items.append((kk, vv))

    option_letters = "".join([k for k, _ in items])
    options_str = " ".join([f"({k}) {v}" for k, v in items])
    return options_str, (option_letters if option_letters else "ABCD")



def read_jsonl(fp: str) -> List[Dict[str, Any]]:
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
                "answer_idx": (row.get("answer_idx") or "").strip().upper()
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


def read_dataset_any(fp: str, data_type: str) -> List[Dict[str, Any]]:
    """
    Unified loader returns list of dicts with:
      {"question": str, "options": dict/list, "answer_idx": optional}
    """
    if data_type == "medqa_jsonl":
        return read_jsonl(fp)
    if data_type == "medbullets_csv":
        return read_medbullets_csv(fp)
    if data_type == "medbullets_json":
        return read_medbullets_json_columnwise(fp)
    raise ValueError(f"Unsupported data_type={data_type}")


def get_terminators(tokenizer) -> List[int]:
    terminators: List[int] = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))
  
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0 and eot not in terminators:
            terminators.append(int(eot))
    except Exception:
        pass
    return list(dict.fromkeys(terminators))


def build_chat_input(tokenizer, question: str, options_str: str, prompt_type: str) -> Dict[str, torch.Tensor]:
    system_msg = USMLE_SYSTEM_COT if prompt_type == "cot" else USMLE_SYSTEM_SP
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": (question + " " + options_str).strip()},
    ]

    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc


@torch.inference_mode()
def generate_once(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    terminators: List[int],
    pad_id: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    outputs = model.generate(
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
    gen = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def infer_file(
    model,
    tokenizer,
    data_type: str,
    data_path: str,
    save_dir: str,
    prompt_type: str,
    max_new_tokens: int = 1024,
    num_samples: int = 1,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 1.0,
    seed: int = 1234,
):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(data_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")
    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_all = os.path.join(save_dir, f"{split}_res_all.jsonl")  # only when SC

    data = read_dataset_any(data_path, data_type=data_type)
    terminators = get_terminators(tokenizer)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    use_sc = (prompt_type == "cot" and num_samples > 1)
    f_all = open(out_all, "w", encoding="utf-8") if use_sc else None

    with open(out_txt, "w", encoding="utf-8") as fraw, open(out_pred, "w", encoding="utf-8") as fpred:
        for i, ex in enumerate(tqdm(data, desc=f"Llama31 {prompt_type.upper()} {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str, option_letters = format_options_parentheses(raw_options)

            enc = build_chat_input(tokenizer, q, options_str, prompt_type=prompt_type)

           
            if (prompt_type == "sp") or (not use_sc):
                text = generate_once(
                    model, tokenizer, enc, terminators, pad_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                )
                letter = extract_answer_letter(text, options=option_letters)
                fraw.write(text.replace("\n", " ") + "\n")
                fpred.write(letter + "\n")
                continue

      
            samples_text: List[str] = []
            samples_letter: List[str] = []

            for s in range(num_samples):
                set_seed(seed + i * 10000 + s)
                t = generate_once(
                    model, tokenizer, enc, terminators, pad_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                l = extract_answer_letter(t, options=option_letters)
                samples_text.append(t)
                samples_letter.append(l)

            winner, counts = majority_vote(samples_letter, valid_letters=option_letters)

            chosen_text = samples_text[0]
            for t, l in zip(samples_text, samples_letter):
                if l == winner:
                    chosen_text = t
                    break

            fraw.write(chosen_text.replace("\n", " ") + "\n")
            fpred.write(winner + "\n")

            if f_all is not None:
                rec = {
                    "idx": i,
                    "question": q,
                    "options_str": options_str,
                    "option_letters": option_letters,
                    "answer_idx": ex.get("answer_idx"),
                    "samples_letter": samples_letter,
                    "vote_counts": counts,
                    "final_letter": winner,
                    "samples_text": samples_text,
                }
                f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if f_all is not None:
        f_all.close()
        print("Saved all samples:", out_all)

    print("Saved raw :", out_txt)
    print("Saved pred:", out_pred)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="local path or HF id")
    p.add_argument("--data_type", type=str, required=True,
                   choices=["medqa_jsonl", "medbullets_csv", "medbullets_json"])
    p.add_argument("--data_path", type=str, required=True, help="a file or a directory")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--prompt_type", type=str, default="sp", choices=["sp", "cot"])
    p.add_argument("--max_new_tokens", type=int, default=1024)

    p.add_argument("--num_samples", type=int, default=1, help="K samples; >1 enables SC for CoT")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
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
            data_type=args.data_type,
            data_path=fp,
            save_dir=args.save_dir,
            prompt_type=args.prompt_type,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

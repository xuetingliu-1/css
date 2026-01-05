
import os
import json
import argparse
import re
from typing import Dict, Any, List, Tuple

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


def extract_answer_letter(text: str, options: str = "ABCDE") -> str:
  
    m = re.findall(rf"the answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.findall(rf"answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

   
    m = re.search(rf"\b([{options}])\b", text)
    return m.group(1).upper() if m else "Z"


def format_options_parentheses(raw_options) -> Tuple[str, str]:
    """
    Return:
      options_str: "(A) xxx (B) yyy ..."
      option_letters: "ABCD" or "ABCDE"
    """
    items: List[Tuple[str, str]] = []

    if isinstance(raw_options, dict):
        for k in sorted(raw_options.keys()):
            kk = str(k).strip().upper()
            if kk in "ABCDE":
                vv = raw_options[k]
                if vv is None:
                    vv = ""
                items.append((kk, str(vv)))
    elif isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            k = chr(ord("A") + i)
            if k in "ABCDE":
                if v is None:
                    v = ""
                items.append((k, str(v)))

    items = [(k, v.strip()) for k, v in items if isinstance(v, str) and v.strip() != ""]

    option_letters = "".join([k for k, _ in items])
    options_str = " ".join([f"({k}) {v}" for k, v in items])
    return options_str, (option_letters if option_letters else "ABCD")


def read_medqa_jsonl(fp: str) -> List[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
    # unique preserve order
    terminators = list(dict.fromkeys(terminators))
    return terminators


def build_chat_input(tokenizer, question: str, options_str: str, prompt_type: str) -> torch.Tensor:
    if prompt_type == "cot":
        system_msg = USMLE_SYSTEM_COT
    else:
        system_msg = USMLE_SYSTEM_SP

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question + " " + options_str},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def infer_file(
    model,
    tokenizer,
    jsonl_path: str,
    save_dir: str,
    prompt_type: str,
    max_new_tokens: int = 1024,
):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(jsonl_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")      # raw generations
    out_pred = os.path.join(save_dir, f"{split}_pred.txt")    # extracted letters

    data = read_medqa_jsonl(jsonl_path)
    terminators = get_terminators(tokenizer)

 
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with open(out_txt, "w", encoding="utf-8") as fraw, open(out_pred, "w", encoding="utf-8") as fpred:
        for i, ex in enumerate(tqdm(data, desc=f"Meerkat {prompt_type.upper()} {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str, option_letters = format_options_parentheses(raw_options)

            input_ids = build_chat_input(tokenizer, q, options_str, prompt_type=prompt_type).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # standard (greedy)
                    temperature=0.0,
                    eos_token_id=terminators, # allow eos or <|eot_id|> to stop
                    pad_token_id=pad_id,
                )

            gen = outputs[0][input_ids.shape[-1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()

            # raw
            fraw.write(text.replace("\n", " ") + "\n")

            # extracted
            letter = extract_answer_letter(text, options=option_letters)
            fpred.write(letter + "\n")

    print("Saved raw :", out_txt)
    print("Saved pred:", out_pred)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="local path or HF id")
    p.add_argument("--data_path", type=str, required=True, help=".jsonl file or a directory of .jsonl")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--prompt_type", type=str, default="sp", choices=["sp", "cot"])
    p.add_argument("--max_new_tokens", type=int, default=1024)
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

    if os.path.isfile(args.data_path) and args.data_path.endswith(".jsonl"):
        files = [args.data_path]
    else:
        files = [
            os.path.join(args.data_path, fn)
            for fn in os.listdir(args.data_path)
            if fn.endswith(".jsonl")
        ]
        files.sort()

    for fp in files:
        infer_file(
            model,
            tokenizer,
            fp,
            args.save_dir,
            prompt_type=args.prompt_type,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

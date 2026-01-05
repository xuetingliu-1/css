

import os
import json
import re
import csv  
import argparse
from typing import Dict, Any, List, Optional
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows



def read_medbullets_csv(path: str) -> List[Dict[str, Any]]:
    """
    MedBullets CSV columns:
      link,question,opa,opb,opc,opd,(ope),answer_idx, ...
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
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

            ans = (row.get("answer_idx") or "").strip().upper()
            rows.append({
                "question": q,
                "options": opts,
                "answer_idx": ans,
            })
    return rows



def load_data_auto(path: str) -> List[Dict[str, Any]]:
    lp = path.lower()
    if lp.endswith(".jsonl"):
        return read_jsonl(path)
    if lp.endswith(".csv"):
        return read_medbullets_csv(path)
    raise ValueError(f"[ERROR] Unsupported data file: {path} (expect .jsonl or .csv)")


def format_options(raw_options) -> str:
  
    if isinstance(raw_options, dict):
        keys = sorted([str(k).strip().upper() for k in raw_options.keys()], key=lambda x: x)
        lines = []
        for k in keys:
            if k not in LETTER_5:
                continue
            v = raw_options.get(k, "")
            lines.append(f"{k}. {str(v).strip()}")
        return "\n".join(lines).strip()

    if isinstance(raw_options, list):
        lines = []
        for i, v in enumerate(raw_options):
            k = chr(ord("A") + i)
            lines.append(f"{k}. {str(v).strip()}")
        return "\n".join(lines).strip()

    return ""


def valid_letters_from_options(raw_options) -> List[str]:
    if isinstance(raw_options, dict):
        keys = {str(k).strip().upper() for k in raw_options.keys()}
        return LETTER_5 if "E" in keys else LETTER_4
    if isinstance(raw_options, list):
        return LETTER_5 if len(raw_options) >= 5 else LETTER_4
    return LETTER_5


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
    terminators = list(dict.fromkeys(terminators))
    return terminators


def load_model(model_path: str, torch_dtype: str):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_chat(
    model,
    tokenizer,
    system_inst: str,
    user_query: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    messages = [{"role": "system", "content": system_inst},
                {"role": "user", "content": user_query}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
    else:
        prompt = f"[SYSTEM]\n{system_inst}\n\n[USER]\n{user_query}\n\n[ASSISTANT]\n"
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

    terminators = get_terminators(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        eos_token_id=terminators,
        pad_token_id=pad_id,
        attention_mask=attention_mask,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p, top_k=top_k))

    out_ids = model.generate(input_ids, **gen_kwargs)
    gen_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()



def safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def parse_rationale(raw: str) -> str:
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        r = str(obj.get("rationale", "")).strip()
        if r:
            return r
    return (raw or "").strip()[:800]


def extract_best_letter(raw: str, valid_letters: List[str]) -> str:
    vset = set(valid_letters)
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        b = str(obj.get("best", "")).strip().upper()
        if b in vset:
            return b

    m = re.search(r'"best"\s*:\s*"([A-E])"', raw, flags=re.IGNORECASE)
    if m and m.group(1).upper() in vset:
        return m.group(1).upper()

    hits = re.findall(r"\b([A-E])\b", (raw or "").upper())
    for h in reversed(hits):
        if h in vset:
            return h
    return valid_letters[0]



def shuffled_order(valid_letters: List[str], seed: int) -> List[str]:
    import random
    r = random.Random(seed)
    out = valid_letters[:]
    r.shuffle(out)
    return out


def majority_vote(letters: List[str], valid_letters: List[str]) -> str:
    filt = [x for x in letters if x in valid_letters]
    if not filt:
        return valid_letters[0]
    cnt = Counter(filt)
    top = max(cnt.values())
    tied = sorted([k for k, v in cnt.items() if v == top])
    for k in valid_letters:
        if k in tied:
            return k
    return tied[0]


def build_agent_system(assumed_letter: str) -> str:
    return (
        "You are a medical reasoning assistant.\n"
        "You will be given a multiple-choice medical question.\n"
        f"IMPORTANT: For this response, you MUST assume option {assumed_letter} is correct.\n"
        "Write a concise rationale (2-5 sentences) that would make that option correct.\n"
        "Do NOT mention you are assuming.\n"
        "Do NOT pick other options.\n\n"
        "Output MUST be ONE line JSON with keys exactly:\n"
        "{\"assumed\":\"X\",\"rationale\":\"...\"}\n"
    )


def build_agent_user(question: str, options_str: str, assumed_letter: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{options_str.strip()}\n\n"
        f"Assumed correct option: {assumed_letter}\n"
        "Now write rationale JSON."
    )


def build_judge_system() -> str:
    return (
        "You are a strict medical QA judge.\n"
        "IMPORTANT:\n"
        "- Option labels (A/B/C/...) are arbitrary. DO NOT use scenario display order as a signal.\n"
        "- Evaluate each scenario independently with the SAME rubric, then decide.\n\n"
        "You will be given:\n"
        "1) the original multiple-choice question with options\n"
        "2) several scenario rationales, one per option, each trying to justify that option\n\n"
        "Task: Decide the single BEST answer option for the original question.\n"
        "You may criticize scenarios; do not assume they are correct.\n\n"
        "Output MUST be ONE line JSON with keys exactly:\n"
        "{\"best\":\"X\",\"reason\":\"...\"}\n"
    )


def build_judge_user(
    question: str,
    options_str: str,
    scenarios: Dict[str, Any],
    valid_letters: List[str],
    display_order: Optional[List[str]] = None,
    judge_prefill: str = "",
) -> str:
    order = display_order[:] if display_order else valid_letters[:]

    lines = []
    if (judge_prefill or "").strip():
        lines.append((judge_prefill or "").strip())
        lines.append("")

    lines.append("Original question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    lines.append(options_str.strip())
    lines.append("")
    lines.append("Scenarios (each supports one option):")
    for L in order:
        rat = ""
        if isinstance(scenarios.get(L), dict):
            rat = str(scenarios[L].get("rationale", "")).strip()
        lines.append(f"\n[{L}] {rat}")
    lines.append("\nNow answer the original question. Return judge JSON.")
    return "\n".join(lines)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_model_path", type=str, required=True)
    p.add_argument("--judge_model_path", type=str, default="")  # empty => reuse agent model

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--agent_max_new_tokens", type=int, default=192)
    p.add_argument("--judge_max_new_tokens", type=int, default=192)

    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "sampling"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)

    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--debug_first_n", type=int, default=0)

   
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge_permute_k", type=int, default=1,
                   help="Call judge K times and majority vote.")
    p.add_argument("--judge_permute_scenarios", action="store_true",
                   help="Permute scenario DISPLAY order per judge call (labels unchanged).")
    p.add_argument("--judge_prefill", type=str, default="",
                   help="Optional prefill reminding order-invariance.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

  
    data = load_data_auto(args.data_path)
    if not data:
        raise ValueError(f"[ERROR] empty data: {args.data_path}")

    agent_model, agent_tok = load_model(args.agent_model_path, args.torch_dtype)
    if args.judge_model_path and args.judge_model_path != args.agent_model_path:
        judge_model, judge_tok = load_model(args.judge_model_path, args.torch_dtype)
    else:
        judge_model, judge_tok = agent_model, agent_tok

    do_sample = (args.decode == "sampling")

    scenarios_path = os.path.join(args.save_dir, "scenarios.jsonl")
    judge_out_path = os.path.join(args.save_dir, "judge_results.jsonl")

    judge_sys = build_judge_system()

    correct = 0
    total = 0

    with open(scenarios_path, "w", encoding="utf-8") as f_sc, \
         open(judge_out_path, "w", encoding="utf-8") as f_j:

        for i, ex in enumerate(tqdm(data, desc="Meerkat One-stage Scenarios + OrderBias")):
            question = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices")
            options_str = format_options(raw_options)
            valid_letters = valid_letters_from_options(raw_options)

           
            scenarios: Dict[str, Any] = {}
            for L in valid_letters:
                sys_inst = build_agent_system(L)
                user_q = build_agent_user(question, options_str, L)
                raw = generate_chat(
                    agent_model, agent_tok,
                    system_inst=sys_inst,
                    user_query=user_q,
                    max_new_tokens=args.agent_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                scenarios[L] = {
                    "assumed": L,
                    "rationale": parse_rationale(raw),
                    "raw": raw
                }

            f_sc.write(json.dumps({
                "idx": i,
                "question": question,
                "options": raw_options,
                "answer_idx": ex.get("answer_idx"),
                "valid_letters": valid_letters,
                "scenarios": scenarios,
            }, ensure_ascii=False) + "\n")

         
            K = max(1, int(args.judge_permute_k))
            preds_all: List[str] = []
            orders_all: List[List[str]] = []
            judge_raw_all: List[str] = []

            for k in range(K):
                if args.judge_permute_scenarios:
                    seed_k = args.seed + i * 10000 + 900000 + k * 13
                    order_k = shuffled_order(valid_letters, seed_k)
                else:
                    order_k = valid_letters[:]

                judge_user = build_judge_user(
                    question=question,
                    options_str=options_str,
                    scenarios=scenarios,
                    valid_letters=valid_letters,
                    display_order=order_k,
                    judge_prefill=args.judge_prefill,
                )
                judge_raw = generate_chat(
                    judge_model, judge_tok,
                    system_inst=judge_sys,
                    user_query=judge_user,
                    max_new_tokens=args.judge_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                pred_k = extract_best_letter(judge_raw, valid_letters)

                preds_all.append(pred_k)
                orders_all.append(order_k)
                judge_raw_all.append(judge_raw)

            pred = majority_vote(preds_all, valid_letters)

            gold = (ex.get("answer_idx") or "").strip().upper()
            is_ok = None
            if gold in valid_letters:
                total += 1
                is_ok = (pred == gold)
                correct += int(is_ok)

            judge_reason = ""
            obj = safe_extract_json(judge_raw_all[0] if judge_raw_all else "")
            if isinstance(obj, dict):
                judge_reason = str(obj.get("reason", "")).strip()

            f_j.write(json.dumps({
                "idx": i,
                "pred": pred,
                "preds_all": preds_all,
                "orders_all": orders_all,
                "gold": gold if gold else None,
                "correct": bool(is_ok) if is_ok is not None else None,
                "judge_reason_first": judge_reason,
                "judge_raw_first": judge_raw_all[0] if judge_raw_all else "",
                "judge_raw_all": judge_raw_all,
                "decode": args.decode,
                "judge_permute_k": K,
                "judge_permute_scenarios": bool(args.judge_permute_scenarios),
                "seed": args.seed,
            }, ensure_ascii=False) + "\n")

            if args.debug_first_n and (i + 1) >= args.debug_first_n:
                break

    print("[OK] saved:", scenarios_path)
    print("[OK] saved:", judge_out_path)
    if total > 0:
        print(f"[ACC] {correct}/{total} = {correct/total*100:.2f}%")
    else:
        print("[ACC] no valid gold labels found (answer_idx missing/empty).")


if __name__ == "__main__":
    main()

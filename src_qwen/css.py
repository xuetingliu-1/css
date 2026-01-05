
import os
import json
import re
import csv
import argparse
from typing import Dict, Any, List, Optional
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

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


def read_csv_medbullets(path: str) -> List[Dict[str, Any]]:
    
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            link = (r.get("link") or "").strip()
            ans_idx = (r.get("answer_idx") or "").strip().upper()

            has_ope = ("ope" in r) and (r.get("ope") is not None) and (str(r.get("ope")).strip() != "")
            if has_ope:
                opts = {
                    "A": (r.get("opa") or "").strip(),
                    "B": (r.get("opb") or "").strip(),
                    "C": (r.get("opc") or "").strip(),
                    "D": (r.get("opd") or "").strip(),
                    "E": (r.get("ope") or "").strip(),
                }
            else:
                opts = {
                    "A": (r.get("opa") or "").strip(),
                    "B": (r.get("opb") or "").strip(),
                    "C": (r.get("opc") or "").strip(),
                    "D": (r.get("opd") or "").strip(),
                }

            # drop empty options
            opts = {k: v for k, v in opts.items() if v != ""}

            rows.append({
                "question": q,
                "options": opts,
                "answer_idx": ans_idx if ans_idx else None,
                "meta_info": "medbullets",
                "link": link,
            })
    return rows


def read_data_auto(path: str) -> List[Dict[str, Any]]:
    p = path.lower()
    if p.endswith(".jsonl"):
        return read_jsonl(path)
    if p.endswith(".csv"):
        return read_csv_medbullets(path)
    raise ValueError(f"[ERROR] Unsupported data file type: {path} (expect .jsonl or .csv)")


def format_options(raw_options) -> str:
    if isinstance(raw_options, dict):
        keys = sorted([str(k).strip().upper() for k in raw_options.keys()], key=lambda x: x)
        lines = []
        for k in keys:
            if k not in LETTER_5:
                continue
            lines.append(f"{k}. {str(raw_options.get(k, '')).strip()}")
        return "\n".join(lines).strip()
    if isinstance(raw_options, list):
        lines = []
        for i, v in enumerate(raw_options):
            lines.append(f"{chr(ord('A') + i)}. {str(v).strip()}")
        return "\n".join(lines).strip()
    return ""


def valid_letters_from_options(raw_options) -> List[str]:
    if isinstance(raw_options, dict):
        keys = {str(k).strip().upper() for k in raw_options.keys()}
        return LETTER_5 if "E" in keys else LETTER_4
    if isinstance(raw_options, list):
        return LETTER_5 if len(raw_options) >= 5 else LETTER_4
    return LETTER_5



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
    return list(dict.fromkeys(ids))


def load_model(model_path: str, gpu_id: int, dtype: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_chat(model, tokenizer, system_inst: str, user_query: str, max_new_tokens: int,
                  do_sample: bool, temperature: float, top_p: float, top_k: int) -> str:
    messages = [{"role": "system", "content": system_inst},
                {"role": "user", "content": user_query}]

    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    eos_ids = get_stop_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        eos_token_id=eos_ids,
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


def parse_rationale(raw: str, max_len: int = 800) -> str:
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        r = str(obj.get("rationale", "")).strip()
        if r:
            return r[:max_len]
    return (raw or "").strip()[:max_len]


def extract_best_letter(raw: str, valid_letters: List[str]) -> str:
    vset = set(valid_letters)
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        b = str(obj.get("best", "")).strip().upper()
        if b in vset:
            return b
    m = re.search(r'"best"\s*:\s*"([A-E])"', raw or "", flags=re.IGNORECASE)
    if m and m.group(1).upper() in vset:
        return m.group(1).upper()
    hits = re.findall(r"\b([A-E])\b", (raw or "").upper())
    for h in reversed(hits):
        if h in vset:
            return h
    return valid_letters[0]


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


def shuffled_order(valid_letters: List[str], seed: int) -> List[str]:
    import random
    r = random.Random(seed)
    out = valid_letters[:]
    r.shuffle(out)
    return out


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
        "- Option labels (A/B/C/...) are arbitrary; do NOT use display order as a signal.\n"
        "- Evaluate each scenario independently with the SAME rubric, then decide.\n\n"
        "You will be given:\n"
        "1) the original multiple-choice question with options\n"
        "2) several scenario rationales, one per option, each trying to justify that option\n\n"
        "Task: Decide the single BEST answer option for the original question.\n"
        "You may criticize scenarios; do not assume they are correct.\n\n"
        "Output MUST be ONE line JSON with keys exactly:\n"
        "{\"best\":\"X\",\"reason\":\"...\"}\n"
    )


def build_judge_user_ordered(
    question: str,
    options_str: str,
    scenarios: Dict[str, Any],
    display_order: List[str],
    judge_prefill: str = "",
) -> str:
    lines = []
    lines.append("Original question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    lines.append(options_str.strip())
    lines.append("")
    lines.append("Scenarios (each supports one option):")
    for L in display_order:
        rat = ""
        if isinstance(scenarios.get(L), dict):
            rat = str(scenarios[L].get("rationale", "")).strip()
        lines.append(f"\n[{L}] {rat}")
    lines.append("\nNow answer the original question. Return judge JSON.")
    body = "\n".join(lines)
    if (judge_prefill or "").strip():
        return f"{judge_prefill.strip()}\n\n{body}"
    return body

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_model_path", type=str, required=True)
    p.add_argument("--judge_model_path", type=str, default="")  # empty => reuse agent model

    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--agent_max_new_tokens", type=int, default=192)
    p.add_argument("--judge_max_new_tokens", type=int, default=192)

    p.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge_permute_k", type=int, default=1)
    p.add_argument("--judge_permute_scenarios", action="store_true")
    p.add_argument("--judge_prefill", type=str, default="")

    p.add_argument("--debug_first_n", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    data = read_data_auto(args.data_path)
    if not data:
        raise ValueError(f"[ERROR] empty data: {args.data_path}")

    agent_model, agent_tok = load_model(args.agent_model_path, args.gpu_id, args.dtype)
    if args.judge_model_path and args.judge_model_path != args.agent_model_path:
        judge_model, judge_tok = load_model(args.judge_model_path, args.gpu_id, args.dtype)
    else:
        judge_model, judge_tok = agent_model, agent_tok

    do_sample = (args.mode == "sampling")

    scenarios_path = os.path.join(args.save_dir, "scenarios.jsonl")
    judge_out_path = os.path.join(args.save_dir, "judge_results.jsonl")

    judge_sys = build_judge_system()
    correct = 0
    total = 0

    with open(scenarios_path, "w", encoding="utf-8") as f_sc, \
         open(judge_out_path, "w", encoding="utf-8") as f_j:

        for i, ex in enumerate(tqdm(data, desc="One-stage (Qwen2.5) + OrderBias")):
            question = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices")
            options_str = format_options(raw_options)
            valid_letters = valid_letters_from_options(raw_options)

            # Stage1: scenarios
            scenarios: Dict[str, Any] = {}
            for L in valid_letters:
                sys_inst = build_agent_system(L)
                user_q = build_agent_user(question, options_str, L)
                raw = generate_chat(
                    agent_model, agent_tok, sys_inst, user_q,
                    max_new_tokens=args.agent_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                scenarios[L] = {"assumed": L, "rationale": parse_rationale(raw), "raw": raw}

            f_sc.write(json.dumps({
                "idx": i,
                "question": question,
                "options": raw_options,
                "answer_idx": ex.get("answer_idx"),
                "scenarios": scenarios,
            }, ensure_ascii=False) + "\n")

            # Stage2: judge with K permutations
            K = max(1, int(args.judge_permute_k))
            judge_preds_all: List[str] = []
            judge_raw_all: List[str] = []
            judge_orders_all: List[List[str]] = []

            for k in range(K):
                if args.judge_permute_scenarios:
                    seed_k = args.seed + i * 10000 + 900000 + k * 13
                    order = shuffled_order(valid_letters, seed_k)
                else:
                    order = valid_letters[:]

                judge_user = build_judge_user_ordered(
                    question=question,
                    options_str=options_str,
                    scenarios=scenarios,
                    display_order=order,
                    judge_prefill=args.judge_prefill,
                )

                judge_raw_k = generate_chat(
                    judge_model, judge_tok, judge_sys, judge_user,
                    max_new_tokens=args.judge_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                pred_k = extract_best_letter(judge_raw_k, valid_letters)

                judge_orders_all.append(order)
                judge_raw_all.append(judge_raw_k)
                judge_preds_all.append(pred_k)

            pred = majority_vote(judge_preds_all, valid_letters)

            gold = ex.get("answer_idx")
            is_ok = (gold is not None and str(pred).upper() == str(gold).upper())
            if gold is not None:
                total += 1
                correct += int(is_ok)

            judge_reason = ""
            obj = safe_extract_json(judge_raw_all[0] if judge_raw_all else "")
            if isinstance(obj, dict):
                judge_reason = str(obj.get("reason", "")).strip()

            f_j.write(json.dumps({
                "idx": i,
                "pred": pred,
                "preds_all": judge_preds_all,
                "orders_all": judge_orders_all,
                "gold": gold,
                "correct": bool(is_ok) if gold is not None else None,
                "judge_reason_first": judge_reason,
                "judge_raw_first": judge_raw_all[0] if judge_raw_all else "",
                "judge_raw_all": judge_raw_all,
            }, ensure_ascii=False) + "\n")

            if args.debug_first_n and (i + 1) >= args.debug_first_n:
                break

    print("[OK] saved:", scenarios_path)
    print("[OK] saved:", judge_out_path)
    if total > 0:
        print(f"[ACC] {correct}/{total} = {correct/total*100:.2f}%")
    else:
        print("[ACC] no gold labels found (answer_idx missing).")


if __name__ == "__main__":
    main()

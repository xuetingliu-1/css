
import os
import sys
import json
import csv
import argparse
import re
from typing import Dict, Any, List, Sequence, Optional
from collections import Counter

from tqdm import tqdm

sys.path.append("")
from model import MedS_Llama3

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="")
    p.add_argument("--gpu_id", type=int, default=0)

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--agent_max_new_tokens", type=int, default=128)
    p.add_argument("--judge_max_new_tokens", type=int, default=128)

    # ---- order-bias mitigation ----
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge_permute_k", type=int, default=1,
                   help="Call judge K times and majority vote.")
    p.add_argument("--judge_permute_scenarios", action="store_true",
                   help="Permute scenario DISPLAY order per judge call (labels unchanged).")
    p.add_argument("--judge_prefill", type=str, default="",
                   help="Prepended text to judge query to remind order-invariance (optional).")

    p.add_argument("--debug_first_n", type=int, default=0)
    return p.parse_args()


def read_jsonl_raw(filepath: str) -> Sequence[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows



def read_medbullets_csv(filepath: str) -> List[dict]:
    """
    MedBullets CSV columns (op4/op5):
      link, question, opa, opb, opc, opd, (ope), answer_idx, ...
    """
    rows: List[dict] = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
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


def load_data_auto(data_path: str) -> Sequence[dict]:
    lp = data_path.lower()
    if lp.endswith(".jsonl"):
        return read_jsonl_raw(data_path)
    if lp.endswith(".csv"):
        return read_medbullets_csv(data_path)
    raise ValueError(f"[ERROR] Unsupported data file: {data_path} (expect .jsonl or .csv)")



def options_to_map(raw_options) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if isinstance(raw_options, dict):
        for k, v in raw_options.items():
            kk = str(k).strip().upper()
            if kk in LETTER_5:
                mp[kk] = str(v)
        return mp
    if isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            k = chr(ord("A") + i)
            mp[k] = str(v)
    return mp


def valid_letters_from_options(raw_options) -> List[str]:
    mp = options_to_map(raw_options)
    return LETTER_5 if "E" in mp else LETTER_4


def format_options_block(mp: Dict[str, str], valid_letters: List[str]) -> str:
    return "\n".join([f"{k}: {mp.get(k, '')}" for k in valid_letters]).strip()


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
    # tie-break by canonical order
    for k in valid_letters:
        if k in tied:
            return k
    return tied[0]


# ----- prompts -----
def build_agent_system(letter: str) -> str:
    return (
        "You are a medical reasoning assistant.\n"
        "You will be given a USMLE-style multiple-choice question.\n\n"
        f"CRITICAL: For this response, you MUST assume option {letter} is correct.\n"
        "Write a concise rationale (2-5 sentences) that would make that option correct.\n"
        "Do NOT mention you are assuming.\n"
        "Do NOT choose other options.\n\n"
        "You MUST end with the exact line:\n"
        f"Answer: {letter}\n"
    )


def build_agent_query(question: str, options_block: str, letter: str) -> str:
    return (
        "Question\n"
        f"{question.strip()}\n\n"
        "Options\n"
        f"{options_block.strip()}\n\n"
        f"Assumed correct option: {letter}\n"
        f"Provide short rationale and end with: Answer: {letter}"
    )


def clean_rationale(text: str, letter: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\n?\s*Answer\s*:\s*[A-E]\s*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(rf"^\s*(Option\s*)?{re.escape(letter)}\s*[:\-]\s*", "", s, flags=re.IGNORECASE).strip()
    return s


JUDGE_SYSTEM = (
    "You are a strict judge for USMLE-style multiple-choice questions.\n"
    "IMPORTANT:\n"
    "- Option labels (A/B/C/...) are arbitrary; DO NOT use display order as a signal.\n"
    "- Evaluate each scenario independently with the SAME rubric, then decide.\n\n"
    "You will be given the original question/options and several scenarios.\n"
    "Each scenario tries to justify a different option.\n"
    "Your job is to decide the single best answer to the original question.\n\n"
    "You may briefly justify, but you MUST end with the exact line:\n"
    "Answer: X\n"
    "where X is one of the provided option letters.\n"
)


def build_judge_query(
    question: str,
    options_map: Dict[str, str],
    valid_letters: List[str],
    scenarios: Dict[str, Any],
    display_order: Optional[List[str]] = None,
    judge_prefill: str = "",
) -> str:

 
    options_block = format_options_block(options_map, valid_letters)
    order = display_order[:] if display_order else valid_letters[:]

    sc_lines = []
    for L in order:
        rat = ""
        if isinstance(scenarios.get(L), dict):
            rat = scenarios[L].get("rationale", "")
        else:
            rat = str(scenarios.get(L, ""))
        sc_lines.append(
            f"Scenario {L}\n"
            f"option text: {options_map.get(L, '')}\n"
            f"rationale: {str(rat).strip()}\n"
        )
    scenarios_block = "\n".join(sc_lines).strip()

    body = (
        "Original Question\n"
        f"{question.strip()}\n\n"
        "Options\n"
        f"{options_block}\n\n"
        "Scenarios\n"
        f"{scenarios_block}\n\n"
        "Now answer the original question.\n"
        "End with exactly: Answer: X"
    )

    pref = (judge_prefill or "").strip()
    if pref:
        return pref + "\n\n" + body
    return body


def extract_choice(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    m = re.findall(r"\bAnswer\s*:\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        c = m[-1].upper()
        return c if c in valid_letters else "Z"
    hits = re.findall(r"\b([A-E])\b", text.upper())
    for h in reversed(hits):
        if h in valid_letters:
            return h
    return "Z"


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    
    data = load_data_auto(args.data_path)
    if not data:
        raise ValueError(f"[ERROR] empty data: {args.data_path}")

    sdk = MedS_Llama3(model_path=args.model_name_or_path, gpu_id=args.gpu_id)

    scenarios_path = os.path.join(args.save_dir, "scenarios.jsonl")
    pred_path = os.path.join(args.save_dir, "pred.txt")
    trace_path = os.path.join(args.save_dir, "judge_traces.jsonl")

    correct = 0
    total = 0

    with open(scenarios_path, "w", encoding="utf-8") as fsc, \
         open(pred_path, "w", encoding="utf-8") as fpred, \
         open(trace_path, "w", encoding="utf-8") as ftrace:

        for idx, ex in enumerate(tqdm(data, desc="One-stage (MedS-Ins) + OrderBias")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices")
            options_map = options_to_map(raw_options)
            valid_letters = valid_letters_from_options(raw_options)
            options_block = format_options_block(options_map, valid_letters)

            scenarios: Dict[str, Any] = {}
            for L in valid_letters:
                sys_inst = build_agent_system(L)
                query = build_agent_query(q, options_block, L)
                try:
                    raw = sdk.chat([], query, sys_inst, max_new_tokens=args.agent_max_new_tokens)
                except TypeError:
                    raw = sdk.chat([], query, sys_inst)

                scenarios[L] = {"assumed": L, "rationale": clean_rationale(raw, L), "raw": raw}

            fsc.write(json.dumps({
                "idx": idx,
                "question": q,
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
                    seed_k = args.seed + idx * 10000 + 900000 + k * 13
                    order_k = shuffled_order(valid_letters, seed_k)
                else:
                    order_k = valid_letters[:]

                judge_query = build_judge_query(
                    question=q,
                    options_map=options_map,
                    valid_letters=valid_letters,
                    scenarios=scenarios,
                    display_order=order_k,
                    judge_prefill=args.judge_prefill,
                )

                try:
                    judge_raw = sdk.chat([], judge_query, JUDGE_SYSTEM, max_new_tokens=args.judge_max_new_tokens)
                except TypeError:
                    judge_raw = sdk.chat([], judge_query, JUDGE_SYSTEM)

                pred_k = extract_choice(judge_raw, valid_letters)

                preds_all.append(pred_k)
                orders_all.append(order_k)
                judge_raw_all.append(judge_raw)

            pred = majority_vote(preds_all, valid_letters)
            fpred.write(pred + "\n")

            gold = ex.get("answer_idx")
            is_ok = None
            if gold in valid_letters and pred in valid_letters:
                total += 1
                is_ok = (pred == gold)
                correct += int(is_ok)

            ftrace.write(json.dumps({
                "idx": idx,
                "pred": pred,
                "preds_all": preds_all,
                "orders_all": orders_all,
                "gold": gold,
                "correct": is_ok,
                "judge_raw_first": judge_raw_all[0] if judge_raw_all else "",
                "judge_raw_all": judge_raw_all,
                "valid_letters": valid_letters,
            }, ensure_ascii=False) + "\n")

            if args.debug_first_n > 0 and (idx + 1) >= args.debug_first_n:
                break

    if total > 0:
        acc = correct / total * 100.0
        print(f"[ACC] {acc:.2f}% ({correct}/{total})")
    else:
        print("[WARN] no valid gold labels to compute accuracy.")

    print("[OK] saved:", scenarios_path)
    print("[OK] saved:", pred_path)
    print("[OK] saved:", trace_path)


if __name__ == "__main__":
    main()

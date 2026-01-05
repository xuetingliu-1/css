
import os
import re
import json
import csv
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]



def read_jsonl_keep_order(path: str) -> List[dict]:
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
    rows = read_jsonl_keep_order(path)
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
                    opts_dict[letters[i]] = str(v)
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
            if row.get("opa", "") != "": opts["A"] = row.get("opa", "")
            if row.get("opb", "") != "": opts["B"] = row.get("opb", "")
            if row.get("opc", "") != "": opts["C"] = row.get("opc", "")
            if row.get("opd", "") != "": opts["D"] = row.get("opd", "")
            if row.get("ope", "") != "": opts["E"] = row.get("ope", "")
            gt = str(row.get("answer_idx", "Z")).strip().upper()
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out


def valid_letters_from_options(options: Dict[str, str]) -> List[str]:
    if options.get("E", "").strip() != "":
        return LETTER_5
    return LETTER_4


def format_options_lines(options: Dict[str, str], valid_letters: List[str]) -> str:
    lines = []
    for L in valid_letters:
        v = str(options.get(L, "") or "").strip()
        if v != "":
            lines.append(f"{L}. {v}")
    return "\n".join(lines).strip()


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
        f"Question:\n{(question or '').strip()}\n\n"
        f"Options:\n{(options_str or '').strip()}\n\n"
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
    lines.append((question or "").strip())
    lines.append("")
    lines.append("Options:")
    lines.append((options_str or "").strip())
    lines.append("")
    lines.append("Scenarios (each supports one option):")
    for L in order:
        rat = ""
        if isinstance(scenarios.get(L), dict):
            rat = str(scenarios[L].get("rationale", "")).strip()
        lines.append(f"\n[{L}] {rat}")
    lines.append("\nNow answer the original question. Return judge JSON.")
    return "\n".join(lines).strip()



def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def build_chat_prompts(tokenizer, systems: List[str], users: List[str]) -> List[str]:
    convs = []
    for s, u in zip(systems, users):
        convs.append([
            {"role": "system", "content": s},
            {"role": "user", "content": u},
        ])

    prompts = tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True
    )
    if isinstance(prompts, str):
        prompts = [prompts]
    return prompts


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

  
    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "sampling"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

  
    ap.add_argument("--agent_max_tokens", type=int, default=192)
    ap.add_argument("--judge_max_tokens", type=int, default=192)


    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["auto", "half", "bfloat16", "float16", "float32"])

 
    ap.add_argument("--batch_size", type=int, default=8, help="number of examples per batch")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--debug_first_n", type=int, default=0)

    
    ap.add_argument("--drop_option_e", action="store_true")


    ap.add_argument("--judge_permute_k", type=int, default=1,
                    help="Call judge K times and majority vote.")
    ap.add_argument("--judge_permute_scenarios", action="store_true",
                    help="Permute scenario DISPLAY order per judge call (labels unchanged).")
    ap.add_argument("--judge_prefill", type=str, default="",
                    help="Optional prefill reminding order-invariance.")

    args = ap.parse_args()
    ensure_dir(args.save_dir)

    if args.data_type == "medqa_jsonl":
        data = load_medqa_jsonl(args.data_path)
    else:
        data = load_medbullets_csv(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    stop = ["<|eot_id|>"]


    if args.decode == "greedy":
        agent_sp = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.agent_max_tokens, stop=stop, seed=args.seed, n=1,
        )
      
        judge_sp = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.judge_max_tokens, stop=stop, seed=args.seed, n=1,
        )
    else:
        agent_sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_tokens=args.agent_max_tokens, stop=stop, seed=args.seed, n=1,
        )
        judge_sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_tokens=args.judge_max_tokens, stop=stop, seed=args.seed, n=1,
        )

    scenarios_path = os.path.join(args.save_dir, "scenarios.jsonl")
    judge_out_path = os.path.join(args.save_dir, "judge_results.jsonl")

    total = len(data)
    processed = 0
    correct = 0
    have_gold = 0
    t0 = time.time()

    judge_sys = build_judge_system()

    with open(scenarios_path, "w", encoding="utf-8") as f_sc, \
         open(judge_out_path, "w", encoding="utf-8") as f_j:

        for batch in batch_iter(list(enumerate(data)), args.batch_size):
     
            stage1_systems: List[str] = []
            stage1_users: List[str] = []
            metas: List[Tuple[int, Dict[str, Any], str, List[str], str, Dict[str, str], str]] = []
           

            for idx, ex in batch:
                q = ex.get("question", "") or ""
                opts = dict(ex.get("options", {}) or {})
                if args.drop_option_e:
                    opts.pop("E", None)

                valid_letters = valid_letters_from_options(opts)
                options_str = format_options_lines(opts, valid_letters)

                for L in valid_letters:
                    stage1_systems.append(build_agent_system(L))
                    stage1_users.append(build_agent_user(q, options_str, L))
                    metas.append((idx, ex, L, valid_letters, options_str, opts, q))

            stage1_prompts = build_chat_prompts(tokenizer, stage1_systems, stage1_users)
            stage1_outs = llm.generate(prompts=stage1_prompts, sampling_params=agent_sp)

         
            scenarios_by_idx: Dict[int, Dict[str, Any]] = {}
            valid_by_idx: Dict[int, List[str]] = {}
            opts_by_idx: Dict[int, Dict[str, str]] = {}
            q_by_idx: Dict[int, str] = {}
            options_str_by_idx: Dict[int, str] = {}

            for out, meta in zip(stage1_outs, metas):
                idx, ex, L, valid_letters, options_str, opts, q = meta
                txt = out.outputs[0].text.strip() if out.outputs else ""
                rat = parse_rationale(txt)
                if idx not in scenarios_by_idx:
                    scenarios_by_idx[idx] = {}
                scenarios_by_idx[idx][L] = {"assumed": L, "rationale": rat, "raw": txt}
                valid_by_idx[idx] = valid_letters
                opts_by_idx[idx] = opts
                q_by_idx[idx] = q
                options_str_by_idx[idx] = options_str

          
            for idx, ex in batch:
                f_sc.write(json.dumps({
                    "idx": idx,
                    "question": q_by_idx.get(idx, ""),
                    "options": opts_by_idx.get(idx, {}),
                    "answer_idx": ex.get("answer_idx", "Z"),
                    "valid_letters": valid_by_idx.get(idx, []),
                    "scenarios": scenarios_by_idx.get(idx, {}),
                }, ensure_ascii=False) + "\n")

         
            K = max(1, int(args.judge_permute_k))

            stage2_systems: List[str] = []
            stage2_users: List[str] = []
            stage2_metas: List[Tuple[int, Dict[str, Any], List[str], List[str]]] = []
          

          
            for idx, ex in batch:
                valid_letters = valid_by_idx.get(idx, []) or valid_letters_from_options(opts_by_idx.get(idx, {}))
                q = q_by_idx.get(idx, "")
                options_str = options_str_by_idx.get(idx, "")
                scenarios = scenarios_by_idx.get(idx, {})

                for k in range(K):
                    if args.judge_permute_scenarios:
                        seed_k = args.seed + idx * 10000 + 900000 + k * 13
                        order_k = shuffled_order(valid_letters, seed_k)
                    else:
                        order_k = valid_letters[:]

                    stage2_systems.append(judge_sys)
                    stage2_users.append(build_judge_user(
                        question=q,
                        options_str=options_str,
                        scenarios=scenarios,
                        valid_letters=valid_letters,
                        display_order=order_k,
                        judge_prefill=args.judge_prefill,
                    ))
                    stage2_metas.append((idx, ex, valid_letters, order_k))

            stage2_prompts = build_chat_prompts(tokenizer, stage2_systems, stage2_users)
            stage2_outs = llm.generate(prompts=stage2_prompts, sampling_params=judge_sp)

       
            judge_pack: Dict[int, Dict[str, Any]] = {}
            for out, meta in zip(stage2_outs, stage2_metas):
                idx, ex, valid_letters, order_k = meta
                judge_raw = out.outputs[0].text.strip() if out.outputs else ""
                pred_k = extract_best_letter(judge_raw, valid_letters)

                if idx not in judge_pack:
                    judge_pack[idx] = {"preds_all": [], "orders_all": [], "judge_raw_all": []}
                judge_pack[idx]["preds_all"].append(pred_k)
                judge_pack[idx]["orders_all"].append(order_k)
                judge_pack[idx]["judge_raw_all"].append(judge_raw)

           
            for idx, ex in batch:
                valid_letters = valid_by_idx.get(idx, []) or valid_letters_from_options(opts_by_idx.get(idx, {}))
                gold = str(ex.get("answer_idx", "Z") or "Z").strip().upper()

                preds_all = judge_pack.get(idx, {}).get("preds_all", [])
                orders_all = judge_pack.get(idx, {}).get("orders_all", [])
                judge_raw_all = judge_pack.get(idx, {}).get("judge_raw_all", [])

                pred = majority_vote(preds_all, valid_letters) if preds_all else (valid_letters[0] if valid_letters else "Z")

                is_ok = None
                if gold and gold != "Z":
                    have_gold += 1
                    is_ok = (pred == gold)
                    correct += int(is_ok)

                judge_reason_first = ""
                obj = safe_extract_json(judge_raw_all[0] if judge_raw_all else "")
                if isinstance(obj, dict):
                    judge_reason_first = str(obj.get("reason", "")).strip()

                f_j.write(json.dumps({
                    "idx": idx,
                    "pred": pred,
                    "preds_all": preds_all,
                    "orders_all": orders_all,
                    "gold": gold,
                    "correct": is_ok,
                    "judge_reason_first": judge_reason_first,
                    "judge_raw_first": judge_raw_all[0] if judge_raw_all else "",
                    "judge_raw_all": judge_raw_all,
                    "decode": args.decode,
                    "judge_permute_k": K,
                    "judge_permute_scenarios": bool(args.judge_permute_scenarios),
                    "seed": args.seed,
                }, ensure_ascii=False) + "\n")

                if args.debug_first_n > 0 and idx < args.debug_first_n:
                    print(f"\n--- DEBUG idx={idx} ---")
                    print("PRED:", pred, "GOLD:", gold)
                    print("PREDS_ALL:", preds_all)
                    if orders_all:
                        print("ORDER0:", orders_all[0])
                    if judge_raw_all:
                        print("JUDGE_RAW0:\n", judge_raw_all[0][:900])
                    print("------------------\n", flush=True)

            processed += len(batch)
            if processed % args.log_every == 0 or processed == total:
                dt = time.time() - t0
                speed = processed / dt if dt > 0 else 0.0
                acc = (correct / have_gold * 100.0) if have_gold > 0 else 0.0
                print(f"[{processed}/{total}] {speed:.2f} ex/s | acc {acc:.2f}% ({correct}/{have_gold}) | elapsed {dt/60:.1f} min",
                      flush=True)

    print("[OK] saved:", scenarios_path)
    print("[OK] saved:", judge_out_path)
    if have_gold > 0:
        print(f"[ACC] {correct}/{have_gold} = {correct/have_gold*100:.2f}%")
    else:
        print("[ACC] no gold labels found (answer_idx missing or Z).")


if __name__ == "__main__":
    main()

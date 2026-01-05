
import os
import sys
import json
import argparse
import re
from collections import Counter
from typing import Dict, Any, List, Sequence

from tqdm import tqdm

# reuse your wrapper
sys.path.append("/home/x/xl414/cp/models/MedS-Ins/Inference")
from model_self_refine import MedS_Llama3_SR



def set_global_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(fp: str) -> Sequence[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_options(raw_options) -> str:
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
    # strongest: single letter output
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


def majority_vote(choices: List[str]) -> str:
    valid = [c for c in choices if c in ["A", "B", "C", "D", "E"]]
    if not valid:
        return "Z"
    counter = Counter(valid)
    # tie-break by letter to be deterministic
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]


# -------------------------
# Debate prompts
# -------------------------
BASE_SYSTEM = (
    "Given a medical multiple-choice question and a list of options, "
    "select the single best answer. "
    "In your final output, ONLY provide the letter of the best option "
    "(A, B, C, D, or E) without any additional text."
)

PERSONAS = [
    "You are a cautious clinical diagnostician. Be conservative and safety-focused.",
    "You are an evidence-based internal medicine physician. Prefer guideline-level reasoning.",
    "You are a USMLE-style test-taking tutor. Focus on eliminating distractors efficiently.",
    "You are a clinical pharmacist. Pay attention to medications, contraindications, and interactions.",
    "You are a pathologist. Focus on mechanistic interpretation and classic associations.",
]


def build_system_for_agent(mode: str, agent_id: int, num_agents: int) -> str:
    if mode == "persona":
        persona = PERSONAS[agent_id % len(PERSONAS)]
        return persona + "\n" + BASE_SYSTEM
    # sampling mode: same system for all agents
    return BASE_SYSTEM


def build_debate_query(input_text: str, other_answers: List[str], prompt_style: str) -> str:
    others_block = "\n".join([f"Agent {j+1}: {a}" for j, a in enumerate(other_answers)])
    header = "These are the solutions to the problem from other agents:\n"

    if prompt_style == "short":
        tail = (
            "\n\nBased off the opinion of other agents, can you give an updated response? "
            "Output only A, B, C, D, or E."
        )
    else:
        # long
        tail = (
            "\n\nUsing the opinion of other agents as additional advice, can you give an updated response? "
            "Output only A, B, C, D, or E."
        )

    return f"{input_text}\n\n{header}{others_block}{tail}"


def gen_answer(
    sdk: MedS_Llama3_SR,
    query: str,
    system_inst: str,
    args,
    seed: int,
) -> str:
    if args.mode == "sampling":
        set_global_seed(seed)
        out = sdk.chat(
            [],
            query,
            system_inst,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        # persona/greedy
        out = sdk.chat(
            [],
            query,
            system_inst,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    return extract_choice(out)


def run_debate_one(
    sdk: MedS_Llama3_SR,
    input_text: str,
    args,
    example_idx: int,
) -> Dict[str, Any]:
    num_agents = args.num_agents
    rounds = args.num_rounds

    # store per-round answers
    all_rounds: List[List[str]] = []

    answers = []
    for i in range(num_agents):
        sys_i = build_system_for_agent(args.mode, i, num_agents)
        seed = args.seed + example_idx * 10000 + i * 100 + 0
        a = gen_answer(sdk, input_text, sys_i, args, seed)
        answers.append(a)
    all_rounds.append(answers)

    for r in range(1, rounds + 1):
        new_answers = []
        for i in range(num_agents):
            sys_i = build_system_for_agent(args.mode, i, num_agents)
            others = [answers[j] for j in range(num_agents) if j != i]
            debate_query = build_debate_query(input_text, others, args.debate_prompt)

            seed = args.seed + example_idx * 10000 + i * 100 + r
            a_new = gen_answer(sdk, debate_query, sys_i, args, seed)
            new_answers.append(a_new)

        answers = new_answers
        all_rounds.append(answers)

    final_vote = majority_vote(answers)

    return {
        "round_answers": all_rounds,  # list of lists; round 0..R
        "final_answers": answers,     # answers at last round
        "final_vote": final_vote,
    }


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str,
                   default="/home/x/xl414/cp/models/MMed-Llama-3-8B-EnIns")
    p.add_argument("--gpu_id", type=int, default=0)

    p.add_argument("--data_path", type=str, required=True,
                   help="a .jsonl file or a directory containing .jsonl files")
    p.add_argument("--save_dir", type=str, required=True)

    # debate settings (baseline)
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--debate_prompt", type=str, default="short", choices=["short", "long"])

    # mode A/B
    p.add_argument("--mode", type=str, default="persona", choices=["persona", "sampling"],
                   help="persona: greedy with different personas; sampling: do_sample=True with fixed seeds")

    # sampling params (used only when mode=sampling)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    # output control
    p.add_argument("--max_new_tokens", type=int, default=8,
                   help="generation budget per agent per round; 8 is enough for a single letter")
    p.add_argument("--save_traces", action="store_true",
                   help="save per-example debate traces to jsonl")

    return p.parse_args()


def run_file(sdk: MedS_Llama3_SR, jsonl_path: str, save_dir: str, args):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(jsonl_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = read_jsonl(jsonl_path)

    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    with open(out_txt, "w", encoding="utf-8") as fout:
        for idx, src in enumerate(tqdm(data, desc=f"Debate {base}")):
            q = src.get("question") or src.get("stem") or ""
            raw_options = src.get("options") or src.get("choices")
            options_str = format_options(raw_options)
            input_text = build_input_text(q, options_str)

            result = run_debate_one(sdk, input_text, args, example_idx=idx)
            pred = result["final_vote"]
            fout.write(pred + "\n")

            if ftrace is not None:
                rec = {
                    "idx": idx,
                    "mode": args.mode,
                    "num_agents": args.num_agents,
                    "num_rounds": args.num_rounds,
                    "debate_prompt": args.debate_prompt,
                    "prediction": pred,
                    "round_answers": result["round_answers"],
                }
                ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if ftrace is not None:
        ftrace.close()

    print("Saved:", out_txt)
    if args.save_traces:
        print("Saved:", out_trace)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    sdk = MedS_Llama3_SR(
        model_path=args.model_name_or_path,
        gpu_id=args.gpu_id,
    )

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
        run_file(sdk, fp, args.save_dir, args)


if __name__ == "__main__":
    main()

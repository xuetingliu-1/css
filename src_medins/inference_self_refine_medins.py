

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Any, Sequence, Tuple
from tqdm import tqdm


sys.path.append("")
from model_self_refine import MedS_Llama3_SR


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str,
                   default="/home/x/xl414/cp/models/MMed-Llama-3-8B-EnIns")
    p.add_argument("--gpu_id", type=int, default=0)

    p.add_argument("--data_path", type=str, required=True,
                   help="a .jsonl file or a directory containing .jsonl files")
    p.add_argument("--save_dir", type=str, required=True)


    p.add_argument("--max_iters", type=int, default=4, help="paper often uses up to 4 iterations")
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


    p.add_argument("--save_traces", action="store_true",
                   help="save init/feedback/refine traces as jsonl (recommended)")

    return p.parse_args()



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


# --------- Self-Refine prompts (HF-style + SR-style) ----------
def sr_init_instruction() -> str:
    # HF-style system prompt; force letter-only to simplify evaluation
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


def self_refine_one(
    sdk: MedS_Llama3_SR,
    input_text: str,
    args,
) -> Tuple[str, Dict[str, Any]]:
    trace: Dict[str, Any] = {"init": None, "iters": []}

    # INIT
    y = sdk.chat(
        [],
        input_text,
        sr_init_instruction(),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.init_max_new_tokens,
    ).strip()
    trace["init"] = y

    # ITERATE
    for t in range(args.max_iters):
        fb_query = f"{input_text}\n\nProposed answer:\n{y}\n\nFeedback:\n"
        fb = sdk.chat(
            [],
            fb_query,
            sr_feedback_instruction(),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.fb_max_new_tokens,
        ).strip()

        stop = parse_stop_flag(fb)

        refine_query = f"{input_text}\n\nProposed answer:\n{y}\n\nFeedback:\n{fb}\n\nFinal answer:\n"
        y_new = sdk.chat(
            [],
            refine_query,
            sr_refine_instruction(),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.refine_max_new_tokens,
        ).strip()

        trace["iters"].append({"t": t, "feedback": fb, "stop": stop, "refined": y_new})
        y = y_new

        if args.use_stop_flag and stop:
            break

    return y, trace


def run_file(sdk: MedS_Llama3_SR, jsonl_path: str, save_dir: str, args):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(jsonl_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = read_jsonl(jsonl_path)

    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    with open(out_txt, "w", encoding="utf-8") as fout:
        for idx, src in enumerate(tqdm(data, desc=f"Self-Refine {base}")):
            q = src.get("question") or src.get("stem") or ""
            raw_options = src.get("options") or src.get("choices")
            options_str = format_options(raw_options)

            input_text = build_input_text(q, options_str)

            final_text, trace = self_refine_one(sdk, input_text, args)
            final_letter = extract_choice(final_text)

            fout.write(final_letter + "\n")

            if ftrace is not None:
                rec = {
                    "idx": idx,
                    "final_letter": final_letter,
                    "final_raw": final_text,
                    "trace": trace,
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

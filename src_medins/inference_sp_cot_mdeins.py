import os
import sys
import json
import argparse
from typing import Sequence

from tqdm import tqdm

sys.path.append("")
from model import MedS_Llama3  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="MMed-Llama-3-8B-EnIns local path",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which GPU",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="MedQA .jsonl ",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="sp",
        choices=["sp", "cot"],
        help="sp = standard prompting；cot = chain-of-thought prompting",
    )
    return parser.parse_args()


def build_instruction(prompt_type: str) -> str:
   
    if prompt_type == "sp":
        return (
            "Given a question and a list of options, select the correct answer from the options directly."
        )

    elif prompt_type == "cot":
        return (
            "Given a medical multiple-choice question and a list of options, "
            "first briefly explain your reasoning step by step. "
            "Then, on a new line, provide your final answer strictly in the following format:\n\n"
            "The right answer is X: <option_text>\n\n"
            "where X must be one of A, B, C, D, or E."
        )

    return "Given a question and a list of options, select the correct answer from the options directly."


def read_jsonl(filepath: str) -> Sequence[dict]:
    src_dict_ls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src = json.loads(line)
            src_dict_ls.append(src)

    res = []
    for src in src_dict_ls:
        question = src.get("question") or src.get("stem") or ""

        raw_options = src.get("options") or src.get("choices")
        options_str = ""
        if isinstance(raw_options, dict):
            pieces = []
            for key in sorted(raw_options.keys()):
                pieces.append(f"{key}: {raw_options[key]}")
            options_str = "\t".join(pieces)
        elif isinstance(raw_options, list):
            pieces = []
            for i, content in enumerate(raw_options):
                key = chr(ord("A") + i)
                pieces.append(f"{key}: {content}")
            options_str = "\t".join(pieces)

        res.append(
            {
                "question": question,
                "options_str": options_str,
            }
        )
    return res


def build_query_text(question: str, options_str: str) -> str:
    return f"Question: {question}\nOptions: {options_str}"


def inference_on_file(
    sdk_api: MedS_Llama3,
    instruction: str,
    jsonl_path: str,
    save_dir: str,
):
    data_list = read_jsonl(jsonl_path)

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(jsonl_path)
    split_name = os.path.splitext(base_name)[0]
    save_path = os.path.join(save_dir, f"{split_name}_res.txt")

    with open(save_path, "w", encoding="utf-8") as fout:
        for item in tqdm(data_list, desc=f"Infer {base_name}"):
            query_text = build_query_text(item["question"], item["options_str"])
            response = sdk_api.chat([], query_text, instruction)
            response_clean = response.replace("\n", " ").strip()
            fout.write(response_clean + "\n")

    print(f"Saved: {save_path}")


def main():
    args = parse_args()

    instruction = build_instruction(args.prompt_type)
    print("Prompt type:", args.prompt_type)
    print("INSTRUCTION:\n", instruction, "\n")

    sdk_api = MedS_Llama3(
        model_path=args.model_name_or_path,
        gpu_id=args.gpu_id,
    )

    filepaths = [
        os.path.join(args.data_path, fn)
        for fn in os.listdir(args.data_path)
        if fn.endswith(".jsonl")
    ]
    filepaths.sort()

    for fp in filepaths:
        inference_on_file(sdk_api, instruction, fp, args.save_dir)


if __name__ == "__main__":
    main()
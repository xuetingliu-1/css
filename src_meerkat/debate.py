

import os
import json
import csv
import argparse
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM



def set_global_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def read_jsonl(fp: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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

            rows.append({"question": q, "options": opts})
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
        rows.append({"question": q, "options": opts})
    return rows


def read_dataset_any(fp: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(fp)  # MedQA style dicts
    if ext == ".csv":
        return read_medbullets_csv(fp)
    if ext == ".json":
        return read_medbullets_json_columnwise(fp)
    raise ValueError(f"Unsupported data file: {fp} (expect .jsonl/.csv/.json)")



def format_options_for_input(raw_options) -> Tuple[str, str]:
   
    items: List[Tuple[str, str]] = []

    if isinstance(raw_options, dict):
        for k in sorted(raw_options.keys()):
            kk = str(k).strip().upper()
            if kk in "ABCDE":
                vv = raw_options[k]
                if vv is None:
                    vv = ""
                vv = str(vv).strip()
                if vv:
                    items.append((kk, vv))
    elif isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            kk = chr(ord("A") + i)
            if kk in "ABCDE":
                if v is None:
                    v = ""
                vv = str(v).strip()
                if vv:
                    items.append((kk, vv))

    option_letters = "".join([k for k, _ in items])
    options_str = " ".join([f"({k}) {v}" for k, v in items])
    if not option_letters:
        option_letters = "ABCD"
    return options_str, option_letters


def build_input_text(question: str, options_str: str) -> str:
    return f"Question: {question}\nOptions: {options_str}"



def extract_choice(text: str, valid_letters: str = "ABCDE") -> str:
    """
    Prefer 'Answer: X'. Keep robust fallbacks.
    """
    m = re.search(rf"\bAnswer\s*:\s*([{valid_letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"right answer is\s*([{valid_letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\banswer\s*(is|:)?\s*([{valid_letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()

    m = re.search(rf"\boption\s*([{valid_letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\b([{valid_letters}])\b", text)
    return m.group(1).upper() if m else "Z"


def majority_vote(choices: List[str], valid_letters: str) -> str:
    valid = [c for c in choices if c in valid_letters]
    if not valid:
        return "Z"
    counter = Counter(valid)
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]



BASE_SYSTEM = (
    "You are a medical expert.\n"
    "Given a medical multiple-choice question and a list of options, "
    "think step by step but keep it brief (2-5 sentences).\n"
    "Then output your final choice on a new line in the exact format:\n"
    "Answer: X\n"
    "where X is one of A, B, C, D, or E."
)

PERSONAS = [
    "You are a cautious clinical diagnostician. Be conservative and safety-focused.",
    "You are an evidence-based internal medicine physician. Prefer guideline-level reasoning.",
    "You are a USMLE-style test-taking tutor. Focus on eliminating distractors efficiently.",
    "You are a clinical pharmacist. Pay attention to medications, contraindications, and interactions.",
    "You are a pathologist. Focus on mechanistic interpretation and classic associations.",
]


def build_system_for_agent(mode: str, agent_id: int) -> str:
    if mode == "persona":
        persona = PERSONAS[agent_id % len(PERSONAS)]
        return persona + "\n" + BASE_SYSTEM
    return BASE_SYSTEM


def build_debate_query(
    input_text: str,
    self_prev: str,
    other_prev: List[str],
    prompt_style: str,
) -> str:
    other_block_lines = []
    for j, sol in enumerate(other_prev, start=1):
        other_block_lines.append(f"Agent {j} solution:\n{sol.strip()}\n")
    others_block = "\n".join(other_block_lines).strip()

    header = "These are the solutions to the problem from other agents:\n"
    tail = (
        "\n\nBased off the opinion of other agents, can you give an updated response?"
        if prompt_style == "short"
        else "\n\nUsing the opinion of other agents as additional advice, can you give an updated response?"
    )

    return (
        f"{input_text}\n\n"
        f"Your previous solution:\n{self_prev.strip()}\n\n"
        f"{header}{others_block}"
        f"{tail}\n"
        f"Remember to end with the exact line: Answer: X"
    )


JUDGE_SYSTEM = (
    "You are a strict judge for USMLE-style multiple-choice questions.\n"
    "You will be given the question/options and several agents' solutions.\n"
    "Choose the single best final answer.\n"
    "You may briefly justify, but you MUST end with the exact line:\n"
    "Answer: X\n"
    "where X is one of A, B, C, D, or E."
)


def build_judge_query(input_text: str, agent_solutions: List[str]) -> str:
    blocks = []
    for i, sol in enumerate(agent_solutions, start=1):
        blocks.append(f"Agent {i} solution:\n{sol.strip()}")
    agents_block = "\n\n".join(blocks).strip()

    return (
        f"{input_text}\n\n"
        f"The following are solutions from multiple agents:\n\n"
        f"{agents_block}\n\n"
        f"Select the best answer. End with: Answer: X"
    )


# -------------------------
# HF chat + generation
# -------------------------
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


def ensure_pad_token(tokenizer):
   
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token


def build_chat_input(tokenizer, system_inst: str, user_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
   
    messages = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_text},
    ]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.inference_mode()
def gen_raw(
    model,
    tokenizer,
    system_inst: str,
    user_text: str,
    args,
    seed: int,
    max_new_tokens: int,
) -> str:
    # sampling vs greedy
    if args.mode == "sampling":
        set_global_seed(seed)
        do_sample = True
    else:
        do_sample = False

    terminators = get_terminators(tokenizer)

    ensure_pad_token(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids, attn_mask = build_chat_input(tokenizer, system_inst, user_text)
    input_ids = input_ids.to(model.device)
    attn_mask = attn_mask.to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,   
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=terminators,
        pad_token_id=pad_id,
    )

  
    if do_sample:
        gen_kwargs.update(
            dict(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        )

    outputs = model.generate(**gen_kwargs)
    gen = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text



def run_debate_one(
    model,
    tokenizer,
    input_text: str,
    option_letters: str,
    args,
    example_idx: int,
) -> Dict[str, Any]:
    num_agents = args.num_agents
    rounds = args.num_rounds

    round_solutions: List[List[str]] = []
    round_choices: List[List[str]] = []

    
    sols0, ch0 = [], []
    for i in range(num_agents):
        sys_i = build_system_for_agent(args.mode, i)
        seed = args.seed + example_idx * 10000 + i * 100 + 0
        raw = gen_raw(model, tokenizer, sys_i, input_text, args, seed, max_new_tokens=args.max_new_tokens)
        sols0.append(raw)
        ch0.append(extract_choice(raw, valid_letters=option_letters))
    round_solutions.append(sols0)
    round_choices.append(ch0)

    
    prev_solutions = sols0
    for r in range(1, rounds + 1):
        new_solutions, new_choices = [], []
        for i in range(num_agents):
            sys_i = build_system_for_agent(args.mode, i)
            self_prev = prev_solutions[i]
            others = [prev_solutions[j] for j in range(num_agents) if j != i]

            debate_query = build_debate_query(
                input_text=input_text,
                self_prev=self_prev,
                other_prev=others,
                prompt_style=args.debate_prompt,
            )

            seed = args.seed + example_idx * 10000 + i * 100 + r
            raw = gen_raw(model, tokenizer, sys_i, debate_query, args, seed, max_new_tokens=args.max_new_tokens)
            new_solutions.append(raw)
            new_choices.append(extract_choice(raw, valid_letters=option_letters))

        prev_solutions = new_solutions
        round_solutions.append(new_solutions)
        round_choices.append(new_choices)

    final_solutions = round_solutions[-1]
    final_choices = round_choices[-1]
    final_vote = majority_vote(final_choices, valid_letters=option_letters)

   
    judge_raw = None
    judge_choice = None
    final_pred = final_vote

    if args.use_judge:
        judge_query = build_judge_query(input_text, final_solutions)
        judge_seed = args.seed + example_idx * 10000 + 9999
        judge_raw = gen_raw(
            model, tokenizer,
            JUDGE_SYSTEM,
            judge_query,
            args,
            judge_seed,
            max_new_tokens=args.judge_max_new_tokens,
        )
        judge_choice = extract_choice(judge_raw, valid_letters=option_letters)
        if judge_choice in option_letters:
            final_pred = judge_choice
        else:
            final_pred = final_vote

    return {
        "round_solutions": round_solutions,
        "round_choices": round_choices,
        "final_vote": final_vote,
        "final_choices": final_choices,
        "use_judge": args.use_judge,
        "judge_raw": judge_raw,
        "judge_choice": judge_choice,
        "prediction": final_pred,
    }




def run_file(model, tokenizer, data_path: str, save_dir: str, args):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(data_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")          # 1 letter per line
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")   # optional traces

    data = read_dataset_any(data_path)
    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    with open(out_txt, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(tqdm(data, desc=f"Meerkat Debate {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str, option_letters = format_options_for_input(raw_options)

            input_text = build_input_text(q, options_str)

            result = run_debate_one(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                option_letters=option_letters,
                args=args,
                example_idx=idx,
            )
            pred = result["prediction"]
            if pred not in option_letters:
                pred = "Z"

            fout.write(pred + "\n")

            if ftrace is not None:
                rec = {
                    "idx": idx,
                    "data_file": base,
                    "mode": args.mode,
                    "num_agents": args.num_agents,
                    "num_rounds": args.num_rounds,
                    "debate_prompt": args.debate_prompt,
                    "option_letters": option_letters,
                    "question": q,
                    "options_str": options_str,
                    "prediction": pred,
                    "final_vote": result["final_vote"],
                    "final_choices": result["final_choices"],
                    "use_judge": result["use_judge"],
                    "judge_choice": result["judge_choice"],
                    "judge_raw": result["judge_raw"],
                    "round_choices": result["round_choices"],
                    "round_solutions": result["round_solutions"],
                }
                ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if ftrace is not None:
        ftrace.close()

    print("Saved:", out_txt)
    if args.save_traces:
        print("Saved:", out_trace)



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True, help="local path or HF id")
    p.add_argument("--data_path", type=str, required=True,
                   help="file (.jsonl/.csv/.json) or a directory containing them")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--debate_prompt", type=str, default="short", choices=["short", "long"])

    p.add_argument("--mode", type=str, default="persona", choices=["persona", "sampling"],
                   help="persona: greedy with personas; sampling: do_sample=True with fixed seeds")

    # sampling params (used only when mode=sampling)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--max_new_tokens", type=int, default=192,
                   help="generation budget per agent per round")
    p.add_argument("--use_judge", action="store_true")
    p.add_argument("--judge_max_new_tokens", type=int, default=128)

    p.add_argument("--save_traces", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # file or directory
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
        run_file(model, tokenizer, fp, args.save_dir, args)


if __name__ == "__main__":
    main()


import os
import json
import csv
import re
import argparse
from collections import Counter
from typing import Dict, Any, List, Sequence, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_global_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def chunks(lst: List[int], bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]



def read_jsonl(fp: str) -> Sequence[Dict[str, Any]]:
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
                v = "" if v is None else str(v).strip()
                if v != "":
                    opts[letter] = v
            rows.append({
                "question": q,
                "options": opts,
                "answer_idx": (row.get("answer_idx") or "").strip().upper() if row.get("answer_idx") is not None else ""
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
            v = "" if v is None else str(v).strip()
            if v != "":
                opts[letter] = v

        ans_col = obj.get("answer_idx", {})
        ans = ""
        if isinstance(ans_col, dict):
            ans = str(ans_col.get(ik, "")).strip().upper()

        rows.append({"question": q, "options": opts, "answer_idx": ans})
    return rows


def load_dataset(data_type: str, path: str) -> List[Dict[str, Any]]:
    if data_type == "medqa_jsonl":
        raw = read_jsonl(path)
        out: List[Dict[str, Any]] = []
        for r in raw:
            q = r.get("question") or r.get("stem") or ""
            opts = r.get("options") or r.get("choices") or {}
            opts_dict: Dict[str, str] = {}

            if isinstance(opts, dict):
                for k in sorted(opts.keys()):
                    kk = str(k).strip().upper()
                    if kk in ["A", "B", "C", "D", "E"]:
                        vv = opts.get(k, "")
                        vv = "" if vv is None else str(vv).strip()
                        if vv != "":
                            opts_dict[kk] = vv
            elif isinstance(opts, list):
                for i, v in enumerate(opts[:5]):
                    kk = chr(ord("A") + i)
                    vv = "" if v is None else str(v).strip()
                    if vv != "":
                        opts_dict[kk] = vv

            out.append({
                "question": str(q),
                "options": opts_dict,
                "answer_idx": (r.get("answer_idx") or "").strip().upper() if r.get("answer_idx") is not None else ""
            })
        return out

    if data_type == "medbullets_csv":
        return read_medbullets_csv(path)

    if data_type == "medbullets_json":
        return read_medbullets_json_columnwise(path)

    raise ValueError(f"Unsupported data_type={data_type}")


def format_options_medins(raw_options) -> str:
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
    if not text:
        return "Z"

    m = re.search(r"\bAnswer\s*:\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"right answer is\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\banswer\s*(is|:)?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()

    m = re.search(r"\boption\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    return "Z"


def majority_vote(choices: List[str]) -> str:
    valid = [c for c in choices if c in ["A", "B", "C", "D", "E"]]
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


def build_debate_query(input_text: str, self_prev: str, other_prev: List[str], prompt_style: str) -> str:
    other_block_lines = []
    for j, sol in enumerate(other_prev, start=1):
        sol_clean = sol.strip()
        other_block_lines.append(f"Agent {j} solution:\n{sol_clean}\n")
    others_block = "\n".join(other_block_lines).strip()

    header = "These are the solutions to the problem from other agents:\n"
    if prompt_style == "short":
        tail = "\n\nBased off the opinion of other agents, can you give an updated response?"
    else:
        tail = "\n\nUsing the opinion of other agents as additional advice, can you give an updated response?"

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



def get_terminators_qwen(tokenizer) -> List[int]:
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


def slice_to_max_len(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], max_len: int):
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    input_ids = input_ids[:, -max_len:]
    if attention_mask is not None:
        attention_mask = attention_mask[:, -max_len:]
    return input_ids, attention_mask


def build_messages(system_inst: str, user_query: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_inst}, {"role": "user", "content": user_query}]


@torch.inference_mode()
def generate_batch_qwen(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    terminators: List[int],
    pad_id: int,
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[str]:

    enc = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, max_input_tokens)

    out = model.generate(
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

   
    if attention_mask is not None:
        prompt_lens = attention_mask.sum(dim=1).tolist()
    else:
        prompt_lens = [input_ids.shape[1]] * input_ids.shape[0]

    texts: List[str] = []
    for i in range(out.shape[0]):
        gen_ids = out[i][int(prompt_lens[i]):]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def gen_raw_single(
    model,
    tokenizer,
    query: str,
    system_inst: str,
    terminators: List[int],
    pad_id: int,
    args,
    seed: int,
    max_new_tokens: int,
) -> str:
    if args.mode == "sampling":
        set_global_seed(seed)
        do_sample = True
        temperature = args.temperature
        top_k = args.top_k
        top_p = args.top_p
    else:
        do_sample = False
        temperature = 0.0
        top_k = 0
        top_p = 1.0

    outs = generate_batch_qwen(
        model=model,
        tokenizer=tokenizer,
        messages_list=[build_messages(system_inst, query)],
        terminators=terminators,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return outs[0].strip()



def run_debate_one(
    model,
    tokenizer,
    terminators: List[int],
    pad_id: int,
    input_text: str,
    args,
    example_idx: int,
) -> Dict[str, Any]:
    num_agents = args.num_agents
    rounds = args.num_rounds

    round_solutions: List[List[str]] = []
    round_choices: List[List[str]] = []

    # Round 0
    sols0, ch0 = [], []
    for i in range(num_agents):
        sys_i = build_system_for_agent(args.mode, i)
        seed = args.seed + example_idx * 10000 + i * 100 + 0
        raw = gen_raw_single(model, tokenizer, input_text, sys_i, terminators, pad_id, args, seed, args.max_new_tokens)
        sols0.append(raw)
        ch0.append(extract_choice(raw))
    round_solutions.append(sols0)
    round_choices.append(ch0)

    # Debate rounds
    prev_solutions = sols0
    for r in range(1, rounds + 1):
        new_solutions, new_choices = [], []
        for i in range(num_agents):
            sys_i = build_system_for_agent(args.mode, i)
            self_prev = prev_solutions[i]
            others = [prev_solutions[j] for j in range(num_agents) if j != i]
            debate_query = build_debate_query(input_text, self_prev, others, args.debate_prompt)

            seed = args.seed + example_idx * 10000 + i * 100 + r
            raw = gen_raw_single(model, tokenizer, debate_query, sys_i, terminators, pad_id, args, seed, args.max_new_tokens)
            new_solutions.append(raw)
            new_choices.append(extract_choice(raw))

        prev_solutions = new_solutions
        round_solutions.append(new_solutions)
        round_choices.append(new_choices)

    final_solutions = round_solutions[-1]
    final_choices = round_choices[-1]
    final_vote = majority_vote(final_choices)

    judge_raw = None
    judge_choice = None
    final_pred = final_vote

    if args.use_judge:
        judge_query = build_judge_query(input_text, final_solutions)
        judge_seed = args.seed + example_idx * 10000 + 9999
        judge_raw = gen_raw_single(
            model, tokenizer,
            judge_query, JUDGE_SYSTEM,
            terminators, pad_id,
            args, judge_seed,
            args.judge_max_new_tokens
        )
        judge_choice = extract_choice(judge_raw)
        if judge_choice in ["A", "B", "C", "D", "E"]:
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



def run_file_batched(model, tokenizer, terminators, pad_id, data_type: str, fp: str, save_dir: str, args):
    ensure_dir(save_dir)
    base = os.path.basename(fp)
    split = os.path.splitext(base)[0]

    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = load_dataset(data_type, fp)

    # pre-build input_texts
    input_texts: List[str] = []
    raw_qs: List[str] = []
    raw_opts_list: List[Any] = []
    gts: List[str] = []
    for ex in data:
        q = ex.get("question") or ex.get("stem") or ""
        raw_options = ex.get("options") or ex.get("choices") or {}
        options_str = format_options_medins(raw_options)
        input_texts.append(build_input_text(str(q), str(options_str)))
        raw_qs.append(ex.get("question", ""))
        raw_opts_list.append(raw_options)
        gts.append(ex.get("answer_idx", ""))

    n = len(data)
    num_agents = args.num_agents
    rounds = args.num_rounds

  
    round_solutions: List[List[List[str]]] = []
    round_choices: List[List[List[str]]] = []

    sols0_agents: List[List[str]] = []
    ch0_agents: List[List[str]] = []

    for agent_id in range(num_agents):
        sys_i = build_system_for_agent(args.mode, agent_id)
        outs_agent = [""] * n

        all_idx = list(range(n))
        for batch_idx in chunks(all_idx, max(1, args.batch_size)):
           
            if args.mode == "sampling":
                set_global_seed(args.seed + batch_idx[0] * 10000 + agent_id * 100 + 0)

            messages_list = [build_messages(sys_i, input_texts[i]) for i in batch_idx]

            do_sample = (args.mode == "sampling")
            outs = generate_batch_qwen(
                model, tokenizer, messages_list, terminators, pad_id,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 0.0,
                top_k=args.top_k if do_sample else 0,
                top_p=args.top_p if do_sample else 1.0,
            )
            for i, t in zip(batch_idx, outs):
                outs_agent[i] = t.strip()

        sols0_agents.append(outs_agent)
        ch0_agents.append([extract_choice(x) for x in outs_agent])


    round_solutions.append(sols0_agents)
    round_choices.append(ch0_agents)


    for r in range(1, rounds + 1):
        sols_r_agents: List[List[str]] = []
        ch_r_agents: List[List[str]] = []

        for agent_id in range(num_agents):
            sys_i = build_system_for_agent(args.mode, agent_id)
            outs_agent = [""] * n

            all_idx = list(range(n))
            for batch_idx in chunks(all_idx, max(1, args.batch_size)):
                if args.mode == "sampling":
                    set_global_seed(args.seed + batch_idx[0] * 10000 + agent_id * 100 + r)

                messages_list = []
                for i in batch_idx:
                    self_prev = round_solutions[r - 1][agent_id][i]
                    others = [round_solutions[r - 1][j][i] for j in range(num_agents) if j != agent_id]
                    debate_query = build_debate_query(input_texts[i], self_prev, others, args.debate_prompt)
                    messages_list.append(build_messages(sys_i, debate_query))

                do_sample = (args.mode == "sampling")
                outs = generate_batch_qwen(
                    model, tokenizer, messages_list, terminators, pad_id,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 0.0,
                    top_k=args.top_k if do_sample else 0,
                    top_p=args.top_p if do_sample else 1.0,
                )
                for i, t in zip(batch_idx, outs):
                    outs_agent[i] = t.strip()

            sols_r_agents.append(outs_agent)
            ch_r_agents.append([extract_choice(x) for x in outs_agent])

        round_solutions.append(sols_r_agents)
        round_choices.append(ch_r_agents)

    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None
    with open(out_pred, "w", encoding="utf-8") as fout:
        for idx in range(n):
            final_solutions = [round_solutions[-1][a][idx] for a in range(num_agents)]
            final_choices = [round_choices[-1][a][idx] for a in range(num_agents)]
            final_vote = majority_vote(final_choices)

            judge_raw = None
            judge_choice = None
            final_pred = final_vote

            if args.use_judge:
                judge_query = build_judge_query(input_texts[idx], final_solutions)
                if args.mode == "sampling":
                    set_global_seed(args.seed + idx * 10000 + 9999)
                judge_raw = gen_raw_single(
                    model, tokenizer,
                    judge_query, JUDGE_SYSTEM,
                    terminators, pad_id,
                    args, args.seed + idx * 10000 + 9999,
                    args.judge_max_new_tokens
                )
                judge_choice = extract_choice(judge_raw)
                if judge_choice in ["A", "B", "C", "D", "E"]:
                    final_pred = judge_choice

            fout.write(final_pred + "\n")

            if ftrace is not None:
          
                round_solutions_list = [[round_solutions[r][a][idx] for a in range(num_agents)] for r in range(len(round_solutions))]
                round_choices_list = [[round_choices[r][a][idx] for a in range(num_agents)] for r in range(len(round_choices))]

                rec = {
                    "idx": idx,
                    "data_type": data_type,
                    "mode": args.mode,
                    "num_agents": args.num_agents,
                    "num_rounds": args.num_rounds,
                    "debate_prompt": args.debate_prompt,
                    "prediction": final_pred,
                    "final_vote": final_vote,
                    "final_choices": final_choices,
                    "use_judge": args.use_judge,
                    "judge_choice": judge_choice,
                    "judge_raw": judge_raw,
                    "round_choices": round_choices_list,
                    "round_solutions": round_solutions_list,
                    "gt": gts[idx],
                    "question": raw_qs[idx],
                    "options": raw_opts_list[idx],
                }
                ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if ftrace is not None:
        ftrace.close()

    print("Saved:", out_pred)
    if args.save_traces:
        print("Saved:", out_trace)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True,
                   help="local path or HF id, e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_input_tokens", type=int, default=4096,
                   help="truncate long prompts (keep rightmost tokens)")

    p.add_argument("--data_type", type=str, required=True,
                   choices=["medqa_jsonl", "medbullets_csv", "medbullets_json"])
    p.add_argument("--data_path", type=str, required=True,
                   help="a single file (.jsonl/.csv/.json) or a directory containing those files")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--debate_prompt", type=str, default="short", choices=["short", "long"])

    p.add_argument("--mode", type=str, default="persona", choices=["persona", "sampling"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--use_judge", action="store_true")
    p.add_argument("--judge_max_new_tokens", type=int, default=128)
    p.add_argument("--save_traces", action="store_true")

    # speed / dtype
    p.add_argument("--batch_size", type=int, default=8,
                   help="batch size across examples per (round, agent) generation")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    terminators = get_terminators_qwen(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # allow file or directory
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
        run_file_batched(
            model=model,
            tokenizer=tokenizer,
            terminators=terminators,
            pad_id=pad_id,
            data_type=args.data_type,
            fp=fp,
            save_dir=args.save_dir,
            args=args,
        )


if __name__ == "__main__":
    main()

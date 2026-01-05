
import os
import re
import json
import csv
import time
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]


def read_jsonl_keep_order(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                rows.append({})
            else:
                rows.append(json.loads(line))
    return rows

def load_medqa_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    MedQA jsonl fields:
      - question / stem
      - options / choices : dict(A-E) or list
      - answer_idx : 'A'..'E' or 'A'..'D'
    """
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
            letters = LETTER_5 if len(raw_opts) == 5 else LETTER_4
            for i, v in enumerate(raw_opts):
                if i < len(letters) and v is not None and str(v).strip() != "":
                    opts_dict[letters[i]] = str(v).strip()
        elif isinstance(raw_opts, dict):
            for k in LETTER_5:
                if k in raw_opts and raw_opts[k] is not None and str(raw_opts[k]).strip() != "":
                    opts_dict[k] = str(raw_opts[k]).strip()

        gt = str(r.get("answer_idx", "Z")).strip().upper()
        out.append({"question": q, "options": opts_dict, "answer_idx": gt})

    return out

def load_medbullets_csv(path: str) -> List[Dict[str, Any]]:
    """
    CSV columns:
      question, opa, opb, opc, opd, (ope), answer_idx
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
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
            gt = str(row.get("answer_idx", "Z")).strip().upper()
            out.append({"question": q, "options": opts, "answer_idx": gt})
    return out


def format_mcq_ultra(question: str, options: Dict[str, str]) -> str:
    """
    UltraMedical repo style:
      {question}
      A. ...
      B. ...
      ...
    """
    lines: List[str] = []
    q = (question or "").strip()
    if q:
        lines.append(q)
    for k in LETTER_5:
        if k in options and options[k] is not None and str(options[k]).strip() != "":
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines).strip()

def valid_letters_from_options(options: Dict[str, str]) -> List[str]:
    if "E" in options and str(options.get("E", "")).strip() != "":
        return LETTER_5
    return LETTER_4


_RE_ANSWER_LINE = re.compile(r"(?:^|\n)\s*Answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_SO_ANSWER = re.compile(r"\b(?:so\s*,?\s*)?(?:the\s*)?answer\s*(?:is|:)\s*([A-E])\b", re.IGNORECASE)
_RE_FINAL = re.compile(r"\bfinal\s*answer\s*[:\-]?\s*([A-E])\b", re.IGNORECASE)
_RE_PAREN = re.compile(r"\(([A-E])\)")
_RE_SINGLE = re.compile(r"\b([A-E])\b")

def extract_choice(text: str, valid_letters: List[str]) -> str:
    if not text:
        return "Z"
    t = text.strip()

    for pat in (_RE_ANSWER_LINE, _RE_SO_ANSWER, _RE_FINAL):
        m = pat.search(t)
        if m:
            c = m.group(1).upper()
            if c in valid_letters:
                return c

    m = _RE_PAREN.search(t)
    if m:
        c = m.group(1).upper()
        if c in valid_letters:
            return c

    hits = _RE_SINGLE.findall(t.upper())
    for h in reversed(hits):
        if h in valid_letters:
            return h

    return "Z"

def majority_vote(letters: List[str], valid_letters: List[str]) -> str:
    filt = [x for x in letters if x in valid_letters]
    if not filt:
        return "Z"
    cnt = Counter(filt)
    best = cnt.most_common()
    top_freq = best[0][1]
    tied = [k for k, v in best if v == top_freq]
    if len(tied) == 1:
        return tied[0]
   
    for k in valid_letters:
        if k in tied:
            return k
    return tied[0]

BASE_INSTR = (
    "You are a medical expert. "
    "Given a multiple-choice medical question, reason briefly (2-5 sentences), "
    "then output your final choice on a new line in the exact format:\n"
    "Answer: X\n"
    "where X is one of A, B, C, D, or E."
)

PERSONAS = [
    "Persona: You are a cautious clinical diagnostician. Be conservative and safety-focused.",
    "Persona: You are an evidence-based internal medicine physician. Prefer guideline-level reasoning.",
    "Persona: You are a USMLE-style test-taking tutor. Focus on eliminating distractors efficiently.",
    "Persona: You are a clinical pharmacist. Pay attention to medications, contraindications, and interactions.",
    "Persona: You are a pathologist. Focus on mechanistic interpretation and classic associations.",
]

JUDGE_INSTR = (
    "You are a strict judge for USMLE-style multiple-choice questions. "
    "You will be given the question/options and several agents' solutions. "
    "Select the single best answer. "
    "You may briefly justify, but you MUST end with:\n"
    "Answer: X"
)

def build_agent_prefix(mode: str, agent_id: int) -> str:
    if mode == "persona":
        return PERSONAS[agent_id % len(PERSONAS)] + "\n" + BASE_INSTR
    return BASE_INSTR

def build_debate_user(
    input_text: str,
    agent_prefix: str,
    self_prev: str,
    other_prev: List[str],
    prompt_style: str,
) -> str:
    other_block_lines = []
    for j, sol in enumerate(other_prev, start=1):
        other_block_lines.append(f"Other agent {j} solution:\n{sol.strip()}\n")
    others_block = "\n".join(other_block_lines).strip()

    if prompt_style == "short":
        tail = "Based on other agents' opinions, give an updated response."
    else:
        tail = "Using other agents' opinions as advice, give an updated response."

    return (
        f"{agent_prefix}\n\n"
        f"{input_text}\n\n"
        f"Your previous solution:\n{self_prev.strip()}\n\n"
        f"These are the solutions from other agents:\n{others_block}\n\n"
        f"{tail}\n"
        f"Remember to end with the exact line: Answer: X"
    ).strip()

def build_initial_user(input_text: str, agent_prefix: str) -> str:
    return f"{agent_prefix}\n\n{input_text}\n\nRemember to end with the exact line: Answer: X".strip()

def build_judge_user(input_text: str, final_solutions: List[str]) -> str:
    blocks = []
    for i, sol in enumerate(final_solutions, start=1):
        blocks.append(f"Agent {i} solution:\n{sol.strip()}")
    agents_block = "\n\n".join(blocks).strip()
    return (
        f"{JUDGE_INSTR}\n\n"
        f"{input_text}\n\n"
        f"The following are solutions from multiple agents:\n\n"
        f"{agents_block}\n\n"
        f"Select the best answer. End with: Answer: X"
    ).strip()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def apply_chat_template_batch(tokenizer, user_texts: List[str]) -> List[str]:

    convs = [[{"role": "user", "content": ut}] for ut in user_texts]
    prompts = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    if isinstance(prompts, str):
        return [prompts]
    return prompts

def vllm_generate_texts(llm: LLM, prompts: List[str], sampling: SamplingParams) -> List[str]:
    outs = llm.generate(prompts=prompts, sampling_params=sampling)
  
    texts: List[str] = []
    for o in outs:
        if o.outputs:
            texts.append((o.outputs[0].text or "").strip())
        else:
            texts.append("")
    return texts


def run_debate_one(
    llm: LLM,
    tokenizer,
    input_text: str,
    valid_letters: List[str],
    args,
    example_idx: int,
) -> Dict[str, Any]:
    num_agents = args.num_agents
    rounds = args.num_rounds

    round_solutions: List[List[str]] = []
    round_choices: List[List[str]] = []


    if args.mode == "sampling":
        agent_sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            stop=["<|eot_id|>"],
            seed=args.seed + example_idx * 10000, 
            n=1,
        )
    else:
        agent_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            stop=["<|eot_id|>"],
            seed=args.seed + example_idx * 10000,
            n=1,
        )

    # Round 0: independent
    user_texts0: List[str] = []
    for i in range(num_agents):
        agent_prefix = build_agent_prefix(args.mode, i)
        user_texts0.append(build_initial_user(input_text, agent_prefix))

    prompts0 = apply_chat_template_batch(tokenizer, user_texts0)
    sols0 = vllm_generate_texts(llm, prompts0, agent_sampling)
    ch0 = [extract_choice(s, valid_letters) for s in sols0]

    round_solutions.append(sols0)
    round_choices.append(ch0)

    prev_solutions = sols0


    for r in range(1, rounds + 1):
        user_texts_r: List[str] = []
        for i in range(num_agents):
            agent_prefix = build_agent_prefix(args.mode, i)
            self_prev = prev_solutions[i]
            others = [prev_solutions[j] for j in range(num_agents) if j != i]
            user_texts_r.append(build_debate_user(
                input_text=input_text,
                agent_prefix=agent_prefix,
                self_prev=self_prev,
                other_prev=others,
                prompt_style=args.debate_prompt,
            ))
        prompts_r = apply_chat_template_batch(tokenizer, user_texts_r)
        sols_r = vllm_generate_texts(llm, prompts_r, agent_sampling)
        ch_r = [extract_choice(s, valid_letters) for s in sols_r]

        round_solutions.append(sols_r)
        round_choices.append(ch_r)
        prev_solutions = sols_r

    final_solutions = round_solutions[-1]
    final_choices = round_choices[-1]
    final_vote = majority_vote(final_choices, valid_letters)

    judge_raw = None
    judge_choice = None
    final_pred = final_vote

    if args.use_judge:
        judge_sampling = SamplingParams(
            temperature=0.0 if args.judge_greedy else args.judge_temperature,
            top_p=1.0 if args.judge_greedy else args.judge_top_p,
            top_k=-1 if args.judge_greedy else args.judge_top_k,
            max_tokens=args.judge_max_tokens,
            stop=["<|eot_id|>"],
            seed=args.seed + example_idx * 10000 + 9999,
            n=1,
        )
        judge_user = build_judge_user(input_text, final_solutions)
        judge_prompt = apply_chat_template_batch(tokenizer, [judge_user])[0]
        judge_raw = vllm_generate_texts(llm, [judge_prompt], judge_sampling)[0]
        judge_choice = extract_choice(judge_raw, valid_letters)

        if judge_choice in valid_letters:
            final_pred = judge_choice
        else:
            final_pred = final_vote

    return {
        "round_solutions": round_solutions,
        "round_choices": round_choices,
        "final_vote": final_vote,
        "final_choices": final_choices,
        "use_judge": bool(args.use_judge),
        "judge_raw": judge_raw,
        "judge_choice": judge_choice,
        "prediction": final_pred,
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, help="HF id or local path")
    p.add_argument("--data_type", type=str, choices=["medqa_jsonl", "medbullets_csv"], required=True)
    p.add_argument("--data_path", type=str, required=True, help="a file or a directory")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="test")

  
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--debate_prompt", type=str, default="short", choices=["short", "long"])
    p.add_argument("--mode", type=str, default="persona", choices=["persona", "sampling"])

    
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)


    p.add_argument("--max_tokens", type=int, default=256)

    # judge
    p.add_argument("--use_judge", action="store_true")
    p.add_argument("--judge_max_tokens", type=int, default=192)
    p.add_argument("--judge_greedy", action="store_true", help="judge uses greedy decoding")
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=1.0)
    p.add_argument("--judge_top_k", type=int, default=-1)

    # vLLM engine
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "half", "bfloat16", "float16", "float32"])

    # batching/log
    p.add_argument("--batch_size", type=int, default=8, help="batch on examples (not agents).")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_traces", action="store_true", help="save full debate traces (detail.jsonl always saved)")
    p.add_argument("--drop_option_e", action="store_true", help="force 4-option by dropping E if exists")

    return p.parse_args()

def list_files(data_path: str, data_type: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    exts = [".jsonl"] if data_type == "medqa_jsonl" else [".csv"]
    files = []
    for fn in os.listdir(data_path):
        if any(fn.lower().endswith(e) for e in exts):
            files.append(os.path.join(data_path, fn))
    files.sort()
    return files

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    files = list_files(args.data_path, args.data_type)

    for fp in files:
        base = os.path.basename(fp)
        split = os.path.splitext(base)[0] if args.split_name == "test" else args.split_name

        if args.data_type == "medqa_jsonl":
            data = load_medqa_jsonl(fp)
        else:
            data = load_medbullets_csv(fp)

        pred_path = os.path.join(args.out_dir, f"{split}_pred.txt")
        detail_path = os.path.join(args.out_dir, f"{split}_detail.jsonl")

        total = len(data)
        processed = 0
        t0 = time.time()
        print(f"\n=== File: {fp} | total={total} ===", flush=True)

        with open(pred_path, "w", encoding="utf-8") as f_pred, \
             open(detail_path, "w", encoding="utf-8") as f_det:

            for batch in batch_iter(list(enumerate(data)), args.batch_size):
                
                for idx, ex in batch:
                    q = ex.get("question", "")
                    opts = dict(ex.get("options", {}) or {})

                    if args.drop_option_e and "E" in opts:
                        opts.pop("E", None)

                    input_text = format_mcq_ultra(q, opts)
                    valid_letters = valid_letters_from_options(opts)

                    result = run_debate_one(
                        llm=llm,
                        tokenizer=tokenizer,
                        input_text=input_text,
                        valid_letters=valid_letters,
                        args=args,
                        example_idx=idx,
                    )
                    pred = result["prediction"]
                    f_pred.write(pred + "\n")

                    rec = {
                        "idx": idx,
                        "mode": args.mode,
                        "num_agents": args.num_agents,
                        "num_rounds": args.num_rounds,
                        "debate_prompt": args.debate_prompt,
                        "prediction": pred,
                        "gt": ex.get("answer_idx", "Z"),
                        "valid_letters": valid_letters,
                        "question": q,
                        "options": opts,
                        "final_vote": result["final_vote"],
                        "final_choices": result["final_choices"],
                        "use_judge": result["use_judge"],
                        "judge_choice": result["judge_choice"],
                        "judge_raw": result["judge_raw"],
                        "round_choices": result["round_choices"],
                        "round_solutions": result["round_solutions"],
                        "gen": {
                            "max_tokens": args.max_tokens,
                            "temperature": 0.0 if args.mode == "persona" else args.temperature,
                            "top_p": 1.0 if args.mode == "persona" else args.top_p,
                            "top_k": -1 if args.mode == "persona" else args.top_k,
                            "seed_base": args.seed,
                            "stop": ["<|eot_id|>"],
                        },
                        "judge_gen": {
                            "judge_max_tokens": args.judge_max_tokens,
                            "judge_greedy": bool(args.judge_greedy),
                            "judge_temperature": 0.0 if args.judge_greedy else args.judge_temperature,
                            "judge_top_p": 1.0 if args.judge_greedy else args.judge_top_p,
                            "judge_top_k": -1 if args.judge_greedy else args.judge_top_k,
                        } if args.use_judge else None,
                        "drop_option_e": bool(args.drop_option_e),
                    }
                    f_det.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    processed += 1
                    if processed % args.log_every == 0 or processed == total:
                        dt = time.time() - t0
                        speed = processed / dt if dt > 0 else 0.0
                        print(f"[{processed}/{total}] {speed:.2f} ex/s | elapsed {dt/60:.1f} min", flush=True)

        print("Saved:")
        print(" ", pred_path)
        print(" ", detail_path)

if __name__ == "__main__":
    main()

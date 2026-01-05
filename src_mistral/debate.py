
import os
import json
import csv
import re
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Sequence, Optional

import torch
from tqdm import tqdm


try:
    from transformers import FineGrainedFP8Config
    _HAS_FP8CFG = True
except Exception:
    _HAS_FP8CFG = False


try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


try:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    _HAS_MISTRAL3 = True
except Exception:
    _HAS_MISTRAL3 = False



def set_global_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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


def build_debate_query(
    input_text: str,
    self_prev: str,
    other_prev: List[str],
    prompt_style: str,
) -> str:
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


def _require_mistral3():
    if not _HAS_MISTRAL3:
        raise RuntimeError(
            "Missing Mistral3ForConditionalGeneration / MistralCommonBackend.\n"
            "Per the model card, you likely need Transformers 'main' + mistral-common.\n"
            "Try:\n"
            "  pip install git+https://github.com/huggingface/transformers\n"
            "  pip install mistral-common --upgrade\n"
        )


def load_tokenizer(model_path: str):
    _require_mistral3()
    tok = MistralCommonBackend.from_pretrained(model_path)
    if hasattr(tok, "padding_side"):
        tok.padding_side = "right"
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_path: str,
    dtype: Optional[torch.dtype],
    device_map: str,
    load_in_4bit: bool,
    dequantize_fp8_to_bf16: bool,
):
    _require_mistral3()
    kwargs = dict(device_map=device_map)

    if dtype is not None:
        try:
            kwargs["dtype"] = dtype
        except Exception:
            kwargs["torch_dtype"] = dtype

    if load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available, cannot use --load_in_4bit")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(dtype or torch.float16),
        )

    if dequantize_fp8_to_bf16:
        if not _HAS_FP8CFG:
            raise RuntimeError("FineGrainedFP8Config not available; cannot use --dequantize_fp8_to_bf16")
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)

    model = Mistral3ForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


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
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, max_input_tokens)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_k=top_k if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        use_cache=True,
        attention_mask=attention_mask,
    )
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = int(eos_id)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = int(pad_id)

    out = model.generate(input_ids=input_ids, **gen_kwargs)

    if attention_mask is not None:
        prompt_len = int(attention_mask[0].sum().item())
    else:
        prompt_len = input_ids.shape[1]
    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()



def gen_raw(
    model,
    tokenizer,
    query: str,
    system_inst: str,
    args,
    seed: int,
    max_new_tokens: int,
) -> str:
    if args.mode == "sampling":
        set_global_seed(seed)
        out = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=build_messages(system_inst, query),
            max_new_tokens=max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        out = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=build_messages(system_inst, query),
            max_new_tokens=max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
    return out.strip()


def run_debate_one(
    model,
    tokenizer,
    input_text: str,
    args,
    example_idx: int,
) -> Dict[str, Any]:
    num_agents = args.num_agents
    rounds = args.num_rounds

    round_solutions: List[List[str]] = []
    round_choices: List[List[str]] = []

    # Round 0: independent
    sols0, ch0 = [], []
    for i in range(num_agents):
        sys_i = build_system_for_agent(args.mode, i)
        seed = args.seed + example_idx * 10000 + i * 100 + 0
        raw = gen_raw(model, tokenizer, input_text, sys_i, args, seed, max_new_tokens=args.max_new_tokens)
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

            debate_query = build_debate_query(
                input_text=input_text,
                self_prev=self_prev,
                other_prev=others,
                prompt_style=args.debate_prompt,
            )

            seed = args.seed + example_idx * 10000 + i * 100 + r
            raw = gen_raw(model, tokenizer, debate_query, sys_i, args, seed, max_new_tokens=args.max_new_tokens)
            new_solutions.append(raw)
            new_choices.append(extract_choice(raw))

        prev_solutions = new_solutions
        round_solutions.append(new_solutions)
        round_choices.append(new_choices)

    final_solutions = round_solutions[-1]
    final_choices = round_choices[-1]
    final_vote = majority_vote(final_choices)

    # optional judge
    judge_raw = None
    judge_choice = None
    final_pred = final_vote

    if args.use_judge:
        judge_query = build_judge_query(input_text, final_solutions)
        judge_seed = args.seed + example_idx * 10000 + 9999
        judge_raw = gen_raw(
            model, tokenizer,
            judge_query, JUDGE_SYSTEM,
            args, judge_seed,
            max_new_tokens=args.judge_max_new_tokens
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



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True,
                   help="local path to Ministral-3-8B-Instruct-2512")
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

    # sampling params
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--use_judge", action="store_true")
    p.add_argument("--judge_max_new_tokens", type=int, default=128)

    p.add_argument("--save_traces", action="store_true")

    # dtype / quant (like your mistral scripts)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dequantize_fp8_to_bf16", action="store_true")

    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def run_file(model, tokenizer, data_type: str, fp: str, save_dir: str, args):
    ensure_dir(save_dir)
    base = os.path.basename(fp)
    split = os.path.splitext(base)[0]

    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_trace = os.path.join(save_dir, f"{split}_traces.jsonl")

    data = load_dataset(data_type, fp)
    ftrace = open(out_trace, "w", encoding="utf-8") if args.save_traces else None

    with open(out_pred, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(tqdm(data, desc=f"Ministral3 Debate {base}")):
            q = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices") or {}
            options_str = format_options_medins(raw_options)
            input_text = build_input_text(str(q), str(options_str))

            result = run_debate_one(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                args=args,
                example_idx=idx,
            )
            pred = result["prediction"]
            fout.write(pred + "\n")

            if ftrace is not None:
                rec = {
                    "idx": idx,
                    "data_type": data_type,
                    "mode": args.mode,
                    "num_agents": args.num_agents,
                    "num_rounds": args.num_rounds,
                    "debate_prompt": args.debate_prompt,
                    "prediction": pred,
                    "final_vote": result["final_vote"],
                    "final_choices": result["final_choices"],
                    "use_judge": result["use_judge"],
                    "judge_choice": result["judge_choice"],
                    "judge_raw": result["judge_raw"],
                    "round_choices": result["round_choices"],
                    "round_solutions": result["round_solutions"],
                    "gt": ex.get("answer_idx", ""),
                    "question": ex.get("question", ""),
                    "options": raw_options,
                }
                ftrace.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if ftrace is not None:
        ftrace.close()

    print("Saved:", out_pred)
    if args.save_traces:
        print("Saved:", out_trace)


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    dtype: Optional[torch.dtype] = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(
        model_path=args.model_path,
        dtype=dtype,
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        dequantize_fp8_to_bf16=bool(args.dequantize_fp8_to_bf16),
    )

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
        run_file(
            model=model,
            tokenizer=tokenizer,
            data_type=args.data_type,
            fp=fp,
            save_dir=args.save_dir,
            args=args,
        )


if __name__ == "__main__":
    main()

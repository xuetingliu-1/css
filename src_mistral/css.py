
import os
import json
import re
import argparse
import inspect
import csv 
from typing import Dict, Any, List, Optional
from collections import Counter

import torch
from tqdm import tqdm


try:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
except Exception as e:
    raise RuntimeError(
        "Ministral3 requires Transformers (main) + mistral-common.\n"
        "Install:\n"
        "  pip install git+https://github.com/huggingface/transformers\n"
        "  pip install mistral-common --upgrade\n"
        f"Original import error: {e}"
    )

try:
    from transformers import FineGrainedFP8Config
    _HAS_FP8CFG = True
except Exception:
    _HAS_FP8CFG = False


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


def _normalize_answer_idx(x: Any) -> Optional[str]:
    """[MOD] normalize answer_idx from csv/jsonl to 'A'..'E' if possible."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up in LETTER_5:
        return s_up
   
    if s.isdigit():
        v = int(s)
        if 0 <= v <= 4:
            return LETTER_5[v]
        if 1 <= v <= 5:
            return LETTER_5[v - 1]
    return None


def read_data(path: str) -> List[Dict[str, Any]]:

    if path.lower().endswith(".jsonl"):
        return read_jsonl(path)

    if path.lower().endswith(".csv"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                q = (r.get("question") or "").strip()
                # build options from opa/opb/opc/opd/(ope)
                opts: Dict[str, str] = {}
                if r.get("opa") is not None: opts["A"] = str(r.get("opa", "")).strip()
                if r.get("opb") is not None: opts["B"] = str(r.get("opb", "")).strip()
                if r.get("opc") is not None: opts["C"] = str(r.get("opc", "")).strip()
                if r.get("opd") is not None: opts["D"] = str(r.get("opd", "")).strip()
                if r.get("ope") is not None and str(r.get("ope", "")).strip() != "":
                    opts["E"] = str(r.get("ope", "")).strip()

                # normalize gold label
                gold = _normalize_answer_idx(r.get("answer_idx"))

                rows.append({
                    "question": q,
                    "options": opts,
                    "answer_idx": gold,
                    # keep extras but NEVER used in prompts
                    "link": r.get("link"),
                    "answer": r.get("answer"),
                    "explanation": r.get("explanation"),
                })
        return rows

    raise ValueError(f"[ERROR] Unsupported data file suffix (need .jsonl or .csv): {path}")


def format_options(raw_options) -> str:
    if isinstance(raw_options, dict):
        keys = sorted([str(k).strip() for k in raw_options.keys()], key=lambda x: x)
        return "\n".join([f"{k}. {str(raw_options[k]).strip()}" for k in keys]).strip()
    if isinstance(raw_options, list):
        lines = []
        for i, v in enumerate(raw_options):
            lines.append(f"{chr(ord('A')+i)}. {str(v).strip()}")
        return "\n".join(lines).strip()
    return ""


def valid_letters_from_options(raw_options) -> List[str]:
    if isinstance(raw_options, dict):
        keys = {str(k).strip().upper() for k in raw_options.keys()}
        return LETTER_5 if "E" in keys else LETTER_4
    if isinstance(raw_options, list):
        return LETTER_5 if len(raw_options) >= 5 else LETTER_4
    return LETTER_5


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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def majority_vote(letters: List[str], valid_letters: List[str]) -> str:
    filt = [x for x in letters if x in valid_letters]
    if not filt:
        return valid_letters[0]
    cnt = Counter(filt)
    top = max(cnt.values())
    tied = sorted([k for k, v in cnt.items() if v == top])
    for k in valid_letters:  # tie-break by canonical order
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
    display_order: List[str],
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
    return "\n".join(lines)


def parse_rationale(raw: str) -> str:
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        r = str(obj.get("rationale", "")).strip()
        if r:
            return r
    return (raw or "").strip()[:600]


def extract_best_letter(raw: str, valid_letters: List[str]) -> str:
    vset = set(valid_letters)
    obj = safe_extract_json(raw)
    if isinstance(obj, dict):
        b = str(obj.get("best", "")).strip().upper()
        if b in vset:
            return b
    m = re.search(r'"best"\s*:\s*"([A-E])"', raw, flags=re.IGNORECASE)
    if m and m.group(1).upper() in vset:
        return m.group(1).upper()
    hits = re.findall(r"\b([A-E])\b", (raw or "").upper())
    for h in reversed(hits):
        if h in vset:
            return h
    return valid_letters[0]


def _dtype_from_arg(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _from_pretrained_with_dtype_kw(cls, model_path: str, *, device_map: str, torch_dtype: torch.dtype, quantization_config=None):
    sig = inspect.signature(cls.from_pretrained)
    kwargs = {"device_map": device_map}
    if "dtype" in sig.parameters:
        kwargs["dtype"] = torch_dtype
    else:
        kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return cls.from_pretrained(model_path, **kwargs)


def load_ministral3(model_path: str, gpu_id: int, dtype: str, dequantize_fp8_to_bf16: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tok = MistralCommonBackend.from_pretrained(model_path)

    torch_dtype = _dtype_from_arg(dtype)
    quant_cfg = None
    if dequantize_fp8_to_bf16:
        if not _HAS_FP8CFG:
            raise RuntimeError(
                "FineGrainedFP8Config not available in your transformers build.\n"
                "You likely did not install Transformers from main.\n"
                "Install:\n  pip install git+https://github.com/huggingface/transformers"
            )
        quant_cfg = FineGrainedFP8Config(dequantize=True)

    model = _from_pretrained_with_dtype_kw(
        Mistral3ForConditionalGeneration,
        model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )
    model.eval()

    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    return model, tok


def _apply_chat_template(tokenizer, messages):
    fn = tokenizer.apply_chat_template
    sig = inspect.signature(fn)
    kwargs = dict(return_tensors="pt", return_dict=True)
    if "add_generation_prompt" in sig.parameters:
        kwargs["add_generation_prompt"] = True
    tokenized = fn(messages, **kwargs)
    return tokenized


def _move_tokenized_to_device(tokenized: Dict[str, Any], device: torch.device, pixel_dtype: torch.dtype):
    for k, v in list(tokenized.items()):
        if torch.is_tensor(v):
            if k == "pixel_values":
                tokenized[k] = v.to(device=device, dtype=pixel_dtype)
            else:
                tokenized[k] = v.to(device=device)
    return tokenized


@torch.inference_mode()
def generate_chat(
    model,
    tokenizer,
    system_inst: str,
    user_query: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    messages = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_query},
    ]

    tokenized = _apply_chat_template(tokenizer, messages)
    tokenized = _move_tokenized_to_device(tokenized, model.device, pixel_dtype=torch.bfloat16)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ))

    output = model.generate(**tokenized, **gen_kwargs)[0]
    prompt_len = int(tokenized["input_ids"].shape[1])
    gen_ids = output[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_model_path", type=str, required=True)
    p.add_argument("--judge_model_path", type=str, default="")
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

    p.add_argument("--dequantize_fp8_to_bf16", action="store_true")

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge_permute_k", type=int, default=1)
    p.add_argument("--judge_permute_scenarios", action="store_true")
    p.add_argument("--judge_prefill", type=str, default="")

    p.add_argument("--debug_first_n", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_dir)


    data = read_data(args.data_path)
    if not data:
        raise ValueError(f"[ERROR] empty data: {args.data_path}")

    agent_model, agent_tok = load_ministral3(
        args.agent_model_path, args.gpu_id, args.dtype, args.dequantize_fp8_to_bf16
    )
    if args.judge_model_path and args.judge_model_path != args.agent_model_path:
        judge_model, judge_tok = load_ministral3(
            args.judge_model_path, args.gpu_id, args.dtype, args.dequantize_fp8_to_bf16
        )
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

        for i, ex in enumerate(tqdm(data, desc="Ministral3 One-stage")):
            question = ex.get("question") or ex.get("stem") or ""
            raw_options = ex.get("options") or ex.get("choices")
            options_str = format_options(raw_options)
            valid_letters = valid_letters_from_options(raw_options)

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

                judge_user = build_judge_user(question, options_str, scenarios, display_order=order)
                if (args.judge_prefill or "").strip():
                    judge_user = f"{args.judge_prefill.strip()}\n\n{judge_user}"

                judge_raw = generate_chat(
                    judge_model, judge_tok, judge_sys, judge_user,
                    max_new_tokens=args.judge_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                pred_k = extract_best_letter(judge_raw, valid_letters)
                judge_orders_all.append(order)
                judge_raw_all.append(judge_raw)
                judge_preds_all.append(pred_k)

            pred = majority_vote(judge_preds_all, valid_letters)

            gold = ex.get("answer_idx")
            is_ok = (gold is not None and str(pred).upper() == str(gold).upper())
            if gold is not None:
                total += 1
                correct += int(is_ok)

            judge_reason_first = ""
            obj = safe_extract_json(judge_raw_all[0] if judge_raw_all else "")
            if isinstance(obj, dict):
                judge_reason_first = str(obj.get("reason", "")).strip()

            f_j.write(json.dumps({
                "idx": i,
                "pred": pred,
                "preds_all": judge_preds_all,
                "orders_all": judge_orders_all,
                "gold": gold,
                "correct": is_ok if gold is not None else None,
                "judge_reason_first": judge_reason_first,
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

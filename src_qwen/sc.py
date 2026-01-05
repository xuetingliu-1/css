
import os
import re
import json
import csv
import time
import argparse
from typing import Dict, List, Any, Sequence, Tuple
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


LETTER_5 = ["A", "B", "C", "D", "E"]
LETTER_4 = ["A", "B", "C", "D"]


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def extract_option_letter(text: str) -> str:
    """
    Extract A-E from generated response text.
    Matches your MedS-Ins extraction priorities.
    """
    if not text:
        return ""

    m = re.search(r"OPTION\s+([A-E])\s+IS\s+CORRECT", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"So the answer is[:：]?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"[答答案为是][:：]?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    return ""


def read_jsonl_raw(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_medqa_jsonl(path: str, is_with_rationale: bool) -> List[Dict[str, Any]]:
    """
    Return list of medins-style dict:
      {instruction, input, output, rationale, lang}
    """
    src = read_jsonl_raw(path)
    lang = os.path.basename(path).split(".")[0]

    out: List[Dict[str, Any]] = []
    for r in src:
        question = r.get("question") or r.get("stem") or ""
        question = str(question)

        raw_options = r.get("options") or r.get("choices")
        options = ""
        if isinstance(raw_options, dict):
           
            for key, content in raw_options.items():
                options += f"{key}. {content} "
        elif isinstance(raw_options, list):
            for i, content in enumerate(raw_options):
                key = chr(ord("A") + i)
                options += f"{key}. {content} "
        else:
            options = ""

        answer_id = ""
        if "answer_idx" in r:
            ans = r["answer_idx"]
            if isinstance(ans, str):
                answer_id = ans
            elif isinstance(ans, list):
                answer_id = ",".join(map(str, ans))
            elif isinstance(ans, int):
                answer_id = chr(ord("A") + ans)
        elif "answer" in r:
            ans = r["answer"]
            if isinstance(ans, str):
                answer_id = ans
            elif isinstance(ans, int):
                answer_id = chr(ord("A") + ans)

        rationale = r.get("rationale", "")

        if is_with_rationale:
            tmp = {
                "instruction": (
                    f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. "
                    f"Let’s solve this step-by-step.  You should first give the reason in {lang} for your choice. "
                    f"Then you should give the right answer index of the question."
                ),
                "input": f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output": f"{answer_id}",
                "rationale": f"{rationale}",
                "lang": lang,
            }
        else:
            tmp = {
                "instruction": (
                    f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. "
                    f"Answer with the best option directly."
                ),
                "input": f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output": f"{answer_id}",
                "rationale": f"{rationale}",
                "lang": lang,
            }

        out.append(tmp)

    return out


def load_medbullets_csv(path: str, is_with_rationale: bool) -> List[Dict[str, Any]]:
 
    out: List[Dict[str, Any]] = []
    lang = "en"

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("question") or "").strip()

            options = ""
            for col, letter in [("opa", "A"), ("opb", "B"), ("opc", "C"), ("opd", "D"), ("ope", "E")]:
                v = row.get(col, "")
                if v is None:
                    v = ""
                v = str(v).strip()
                if v != "":
                    options += f"{letter}. {v} "

            answer_id = str(row.get("answer_idx", "") or "").strip().upper()
            rationale = "" 

            if is_with_rationale:
                tmp = {
                    "instruction": (
                        f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. "
                        f"Let’s solve this step-by-step.  You should first give the reason in {lang} for your choice. "
                        f"Then you should give the right answer index of the question."
                    ),
                    "input": f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                    "output": f"{answer_id}",
                    "rationale": f"{rationale}",
                    "lang": lang,
                }
            else:
                tmp = {
                    "instruction": (
                        f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. "
                        f"Answer with the best option directly."
                    ),
                    "input": f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                    "output": f"{answer_id}",
                    "rationale": f"{rationale}",
                    "lang": lang,
                }

            out.append(tmp)

    return out


def prepare_data(data_list: Sequence[dict]) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    for _idx in range(len(data_list)):
        data_entry = data_list[_idx]
        data_list[_idx]["sample_id"] = _idx

        data_list[_idx]["pmc_input"] = (
            prompt_input.format_map(data_entry)
            if data_entry.get("input", "") != ""
            else prompt_no_input.format_map(data_entry)
        )
        data_list[_idx]["pmc_output"] = data_entry["output"]
        data_list[_idx]["rationale"] = data_entry["rationale"]
    return data_list


def build_messages(pmc_input: str, use_system: bool) -> List[Dict[str, str]]:

    if use_system:
        system_msg = "You are a helpful assistant."
        return [{"role": "system", "content": system_msg},
                {"role": "user", "content": pmc_input}]
    else:
        return [{"role": "user", "content": pmc_input}]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def slice_to_max_len(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int):
    if input_ids.shape[1] <= max_len:
        return input_ids, attention_mask
    return input_ids[:, -max_len:], attention_mask[:, -max_len:]


def get_stop_ids(tokenizer: AutoTokenizer) -> List[int]:
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


def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def decode_responses(tokenizer, out_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
   
    texts = []
    for i in range(out_ids.shape[0]):
        prompt_len = int(attention_mask[i].sum().item())
        gen_ids = out_ids[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        texts.append(text)
    return texts


def pick_sc_final(sc_logs: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, int]]:
  
    valid_pairs = [(x["response"], x["letter"]) for x in sc_logs if x.get("letter", "") != ""]
    votes = Counter([l for _, l in valid_pairs]) if valid_pairs else Counter()

    if valid_pairs:
        majority_letter, _ = votes.most_common(1)[0]
        chosen = None
        for resp, letter in valid_pairs:
            if letter == majority_letter:
                chosen = resp
                break
        if chosen is None:
            chosen = valid_pairs[0][0]
        return chosen, majority_letter, dict(votes)

  
    return sc_logs[0]["response"], "", dict(votes)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, required=True, help="Qwen/Qwen2.5-7B-Instruct or local path")
    ap.add_argument("--data_type", type=str, required=True, choices=["medqa_jsonl", "medbullets_csv"])
    ap.add_argument("--data_path", type=str, required=True, help="file path")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

    # MedS-Ins style flag
    ap.add_argument("--is_with_rationale", action="store_true", help="CoT prompting in MedS-Ins format")

    # Self-consistency
    ap.add_argument("--self_consistency_k", type=int, default=1)
    ap.add_argument("--sc_temperature", type=float, default=0.5, help="sampling temperature for SC when k>1")

    # batch + length
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1000) 
    ap.add_argument("--max_input_tokens", type=int, default=4096)

    ap.add_argument("--use_system", action="store_true")
    ap.add_argument("--log_every", type=int, default=40)

    # decoding params (match medins sampling: top_k=40)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=1.0)

    # dtype / quant
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    if args.data_type == "medqa_jsonl":
        data = load_medqa_jsonl(args.data_path, is_with_rationale=args.is_with_rationale)
    else:
        data = load_medbullets_csv(args.data_path, is_with_rationale=args.is_with_rationale)
    data = prepare_data(data)

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if args.load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available, cannot use --load_in_4bit")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=(dtype or torch.float16))
        model_kwargs["quantization_config"] = bnb

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    eos_ids = get_stop_ids(tokenizer)

    res_path = os.path.join(args.out_dir, f"{args.split_name}_res.txt")
    pred_path = os.path.join(args.out_dir, f"{args.split_name}_pred.txt")
    detail_path = os.path.join(args.out_dir, f"{args.split_name}_detail.jsonl")

    sc_debug_root = None
    if args.self_consistency_k > 1:
        sc_debug_root = os.path.join(args.out_dir, f"{args.split_name}_sc_debug")
        ensure_dir(sc_debug_root)

    total = len(data)
    processed = 0
    t0 = time.time()
    print(f"Total examples: {total}", flush=True)

    with open(res_path, "w", encoding="utf-8") as f_res, \
         open(pred_path, "w", encoding="utf-8") as f_pred, \
         open(detail_path, "w", encoding="utf-8") as f_det:

     
        if args.self_consistency_k <= 1:
            for batch in batch_iter(list(enumerate(data)), args.batch_size):
                messages_list = []
                metas: List[Tuple[int, Dict[str, Any]]] = []

                for idx, ex in batch:
                    msgs = build_messages(ex["pmc_input"], use_system=args.use_system)
                    messages_list.append(msgs)
                    metas.append((idx, ex))

                enc = tokenizer.apply_chat_template(
                    messages_list,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)

                input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, args.max_input_tokens)

                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )

                texts = decode_responses(tokenizer, out_ids, attention_mask)

                for (idx, ex), text in zip(metas, texts):
                    
                    if "### Response:" in text:
                        response = text.split("### Response:", 1)[1].strip()
                    else:
                        response = text.strip()
                    response = response.replace("\n", "")

                    gold = ex["pmc_output"]
                    rationale = ex["rationale"]

                    
                    letter = extract_option_letter(response) or "Z"

                    f_res.write(f"{response}[SPLIT]{gold}[SPLIT]{rationale}\n")
                    f_pred.write(f"{letter}\n")
                    f_det.write(json.dumps({
                        "idx": idx,
                        "k": 1,
                        "mode": "greedy",
                        "pred_letter": letter,
                        "gold": gold,
                        "response": response,
                    }, ensure_ascii=False) + "\n")

                processed += len(metas)
                if processed % args.log_every == 0 or processed == total:
                    dt = time.time() - t0
                    speed = processed / dt if dt > 0 else 0.0
                    print(f"[{processed}/{total}] {speed:.2f} ex/s | elapsed {dt/60:.1f} min", flush=True)

            print("Saved:")
            print(" ", res_path)
            print(" ", pred_path)
            print(" ", detail_path)
            return


        K = args.self_consistency_k

        for batch in batch_iter(list(enumerate(data)), args.batch_size):
            metas: List[Tuple[int, Dict[str, Any]]] = list(batch)
            batch_size = len(metas)

            sc_logs_per_ex: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

            for k_idx in range(K):
             
                messages_list = []
                for _, ex in metas:
                    messages_list.append(build_messages(ex["pmc_input"], use_system=args.use_system))

                enc = tokenizer.apply_chat_template(
                    messages_list,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)
                input_ids, attention_mask = slice_to_max_len(input_ids, attention_mask, args.max_input_tokens)

                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=args.sc_temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )

                texts = decode_responses(tokenizer, out_ids, attention_mask)

                for i in range(batch_size):
                    text = texts[i]
                    if "### Response:" in text:
                        resp = text.split("### Response:", 1)[1].strip()
                    else:
                        resp = text.strip()
                    resp = resp.replace("\n", "")

                    letter = extract_option_letter(resp)

                    sc_logs_per_ex[i].append({
                        "k": k_idx,
                        "letter": letter,
                        "response": resp,
                    })

          
            for i, (idx, ex) in enumerate(metas):
                gold = ex["pmc_output"]
                rationale = ex["rationale"]
                sample_id = ex["sample_id"]

                final_response, majority_letter, votes = pick_sc_final(sc_logs_per_ex[i])
                final_letter = majority_letter if majority_letter != "" else (extract_option_letter(final_response) or "Z")
                if final_letter == "":
                    final_letter = "Z"

            
                f_res.write(f"{final_response}[SPLIT]{gold}[SPLIT]{rationale}\n")
                f_pred.write(f"{final_letter}\n")
                f_det.write(json.dumps({
                    "idx": idx,
                    "k": K,
                    "mode": "self_consistency",
                    "pred_letter": final_letter,
                    "majority_letter": majority_letter,
                    "votes": votes,
                    "gold": gold,
                    "final_response": final_response,
                }, ensure_ascii=False) + "\n")

               
                debug_obj = {
                    "sample_id": sample_id,
                    "gold": gold,
                    "rationale": rationale,
                    "self_consistency_k": K,
                    "majority_letter": majority_letter,
                    "votes": votes,
                    "samples": sc_logs_per_ex[i],
                    "final_response": final_response,
                }
                debug_path = os.path.join(sc_debug_root, f"sample_{sample_id:04d}.json")
                with open(debug_path, "w", encoding="utf-8") as f_dbg:
                    json.dump(debug_obj, f_dbg, ensure_ascii=False, indent=2)

            processed += batch_size
            if processed % args.log_every == 0 or processed == total:
                dt = time.time() - t0
                speed = processed / dt if dt > 0 else 0.0
                print(f"[{processed}/{total}] {speed:.2f} ex/s | elapsed {dt/60:.1f} min", flush=True)

    print("Saved:")
    print(" ", res_path)
    print(" ", pred_path)
    print(" ", detail_path)
    if sc_debug_root:
        print(" ", sc_debug_root)


if __name__ == "__main__":
    main()

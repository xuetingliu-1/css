
import os
import json
import csv
import re
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional, Sequence

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


USMLE_SYSTEM_COT = (
    "The following is a multiple-choice question about medical knowledge. "
    "Solve this in a step-by-step fashion, starting by summarizing the available information. "
    "Output a single option from the given options as the final answer. "
    "You are strongly required to follow the specified output format; conclude your response "
    "with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"
)


USMLE_SYSTEM_SP = (
    "The following is a multiple-choice question about medical knowledge. "
    "Output a single option from the given options as the final answer. "
    "You are strongly required to follow the specified output format; conclude your response "
    "with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def extract_answer_letter(text: str, options: str = "ABCDE") -> str:
    if not text:
        return "Z"

    m = re.findall(rf"the answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.findall(rf"answer is\s*\(([{options}])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.search(rf"\b([{options}])\b", text)
    return m.group(1).upper() if m else "Z"


def majority_vote(letters: List[str], valid_letters: str) -> Tuple[str, Dict[str, int]]:
    cnt = Counter([x for x in letters if x in valid_letters])
    if not cnt:
        return "Z", {}
    winner = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return winner, dict(cnt)


def format_options_parentheses(raw_options) -> Tuple[str, str]:
    items: List[Tuple[str, str]] = []

    if isinstance(raw_options, dict):
        for k in sorted(raw_options.keys()):
            kk = str(k).strip().upper()
            if kk in "ABCDE":
                vv = raw_options.get(k, "")
                vv = "" if vv is None else str(vv).strip()
                if vv != "":
                    items.append((kk, vv))
    elif isinstance(raw_options, list):
        for i, v in enumerate(raw_options):
            k = chr(ord("A") + i)
            if k in "ABCDE":
                vv = "" if v is None else str(v).strip()
                if vv != "":
                    items.append((k, vv))

    option_letters = "".join([k for k, _ in items])
    options_str = " ".join([f"({k}) {v}" for k, v in items])
    return options_str, (option_letters if option_letters else "ABCD")



def read_jsonl(fp: str) -> List[Dict[str, Any]]:
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
                "answer_idx": (row.get("answer_idx") or "").strip().upper()
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


def read_dataset_any(fp: str, data_type: str) -> List[Dict[str, Any]]:
    if data_type == "medqa_jsonl":
        return read_jsonl(fp)
    if data_type == "medbullets_csv":
        return read_medbullets_csv(fp)
    if data_type == "medbullets_json":
        return read_medbullets_json_columnwise(fp)
    raise ValueError(f"Unsupported data_type={data_type}")


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


def build_messages(question: str, options_str: str, prompt_type: str) -> List[Dict[str, str]]:
    system_msg = USMLE_SYSTEM_COT if prompt_type == "cot" else USMLE_SYSTEM_SP
    user_msg = (question + " " + options_str).strip()  
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _encode_batch(tokenizer, messages_list: List[List[Dict[str, str]]]):
    
    try:
        enc = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        if isinstance(enc, dict) and "input_ids" in enc:
            return enc
    except Exception:
        pass

    encoded = []
    for msgs in messages_list:
        e = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded.append(e)

    input_ids_list = [e["input_ids"][0] for e in encoded]
    attn_list = [e.get("attention_mask", torch.ones_like(e["input_ids"]))[0] for e in encoded]

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    max_input_tokens: int,
) -> List[str]:
    enc = _encode_batch(tokenizer, messages_list)
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

    texts: List[str] = []
    for i in range(out.shape[0]):
        if attention_mask is not None:
            prompt_len = int(attention_mask[i].sum().item())
        else:
            prompt_len = input_ids.shape[1]
        gen_ids = out[i][prompt_len:]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts

def infer_file(
    model,
    tokenizer,
    data_type: str,
    data_path: str,
    save_dir: str,
    prompt_type: str,
    max_new_tokens: int,
    max_input_tokens: int,
    batch_size: int,
    num_samples: int,
    sc_batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(data_path)
    split = os.path.splitext(base)[0]

    out_txt = os.path.join(save_dir, f"{split}_res.txt")
    out_pred = os.path.join(save_dir, f"{split}_pred.txt")
    out_all = os.path.join(save_dir, f"{split}_res_all.jsonl")  # only when SC

    data = read_dataset_any(data_path, data_type=data_type)

    use_sc = (prompt_type == "cot" and num_samples > 1)
    f_all = open(out_all, "w", encoding="utf-8") if use_sc else None

    def batches(indices: List[int], bs: int):
        for st in range(0, len(indices), bs):
            yield indices[st: st + bs]

    indices = list(range(len(data)))

    with open(out_txt, "w", encoding="utf-8") as fraw, open(out_pred, "w", encoding="utf-8") as fpred:
    
        if use_sc:
            for i in tqdm(indices, desc=f"Ministral3 SC {base}"):
                ex = data[i]
                q = ex.get("question") or ex.get("stem") or ""
                raw_options = ex.get("options") or ex.get("choices") or {}
                options_str, option_letters = format_options_parentheses(raw_options)

                msg = build_messages(q, options_str, prompt_type="cot")

                samples_text: List[str] = []
                samples_letter: List[str] = []

                K = int(num_samples)
                chunk = int(sc_batch_size) if sc_batch_size > 0 else K
                chunk = max(1, min(chunk, K))

                base_seed = seed + i * 10000
                produced = 0
                while produced < K:
                    cur = min(chunk, K - produced)
                    set_seed(base_seed + produced) 
                    msgs_list = [msg for _ in range(cur)]

                    texts = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        messages_list=msgs_list,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        max_input_tokens=max_input_tokens,
                    )
                    for t in texts:
                        l = extract_answer_letter(t, options=option_letters)
                        samples_text.append(t)
                        samples_letter.append(l)
                    produced += cur

                winner, counts = majority_vote(samples_letter, valid_letters=option_letters)

                chosen_text = samples_text[0] if samples_text else ""
                for t, l in zip(samples_text, samples_letter):
                    if l == winner:
                        chosen_text = t
                        break

                fraw.write(chosen_text.replace("\n", " ") + "\n")
                fpred.write(winner + "\n")

                if f_all is not None:
                    rec = {
                        "idx": i,
                        "question": q,
                        "options_str": options_str,
                        "option_letters": option_letters,
                        "answer_idx": ex.get("answer_idx"),
                        "samples_letter": samples_letter,
                        "vote_counts": counts,
                        "final_letter": winner,
                        "samples_text": samples_text,
                    }
                    f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

   
        else:
            bs = max(1, int(batch_size))
            for idxs in tqdm(list(batches(indices, bs)), desc=f"Ministral3 {prompt_type.upper()} {base}"):
                msgs_list: List[List[Dict[str, str]]] = []
                metas: List[str] = []  # option_letters

                for i in idxs:
                    ex = data[i]
                    q = ex.get("question") or ex.get("stem") or ""
                    raw_options = ex.get("options") or ex.get("choices") or {}
                    options_str, option_letters = format_options_parentheses(raw_options)
                    msgs_list.append(build_messages(q, options_str, prompt_type=prompt_type))
                    metas.append(option_letters)

                
                texts = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    messages_list=msgs_list,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                    max_input_tokens=max_input_tokens,
                )

                for option_letters, text in zip(metas, texts):
                    letter = extract_answer_letter(text, options=option_letters)
                    fraw.write(text.replace("\n", " ") + "\n")
                    fpred.write(letter + "\n")

    if f_all is not None:
        f_all.close()
        print("Saved all samples:", out_all)

    print("Saved raw :", out_txt)
    print("Saved pred:", out_pred)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="local path or HF id")
    p.add_argument("--data_type", type=str, required=True,
                   choices=["medqa_jsonl", "medbullets_csv", "medbullets_json"])
    p.add_argument("--data_path", type=str, required=True, help="a file or a directory")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--prompt_type", type=str, default="sp", choices=["sp", "cot"])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_input_tokens", type=int, default=4096, help="truncate long prompts (keep rightmost tokens)")

    # batching
    p.add_argument("--batch_size", type=int, default=8, help="batch across questions (effective when not using SC)")
    p.add_argument("--sc_batch_size", type=int, default=40,
                   help="for SC: generate this many samples per chunk (<=num_samples). Reduce if OOM.")

    # SC args
    p.add_argument("--num_samples", type=int, default=1, help="K samples; >1 enables SC for CoT")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    # dtype / quant (align with your mistral SP script)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dequantize_fp8_to_bf16", action="store_true",
                   help="Load FP8 weights then dequantize to BF16 using FineGrainedFP8Config(dequantize=True).")

    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

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
        infer_file(
            model=model,
            tokenizer=tokenizer,
            data_type=args.data_type,
            data_path=fp,
            save_dir=args.save_dir,
            prompt_type=args.prompt_type,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            sc_batch_size=args.sc_batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import Sequence
import re 

try:
    from peft import AutoPeftModel
except ImportError:
    AutoPeftModel = None

from transformers import AutoModelForCausalLM, AutoTokenizer


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
   
    m = re.search(r"OPTION\s+([A-E])\s+IS\s+CORRECT", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    
    m = re.search(r"So the answer is[:：]?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()


    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument(
    '--is_with_rationale',
    action='store_true',
    help='Use rationale (chain-of-thought) prompting'
)
    parser.add_argument('--is_lora', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)  

    parser.add_argument(
        '--self_consistency_k',
        type=int,
        default=1,
        help='Number of CoT samples per question for self-consistency. '
             'If 1, behaves like original single-sample decoding.'
    )
    args = parser.parse_args()
    return args


def inference_on_one(
    input_str_list: Sequence[str],
    model,
    tokenizer,
    temperature: float = 0.0
) -> str:
 
    model_inputs = tokenizer(
        input_str_list,
        return_tensors='pt',
        padding=True,
    )

    input_ids = model_inputs["input_ids"].cuda()
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()


    do_sample = temperature > 0.0

    topk_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        top_k=50,
        do_sample=do_sample,
        temperature=abs(temperature) if temperature > 0 else None,
    )

    output_str_list = tokenizer.batch_decode(topk_output)
    return output_str_list[0]


def read_jsonl(filepath: str, is_with_rationale):
    """Load a .jsonl file into列表，每一行一条样本 dict。"""
    src_dict_ls = []
    lang = os.path.basename(filepath).split(".")[0]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            src_dict = json.loads(line)
            src_dict["lang"] = lang
            src_dict_ls.append(src_dict)

    res_dict_ls = []
    for src_dict in src_dict_ls:
        lang = src_dict.get("lang", "en")

     
        question = src_dict.get("question") or src_dict.get("stem") or ""

 
        options = ""
        raw_options = src_dict.get("options") or src_dict.get("choices")
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
        if "answer_idx" in src_dict:
            ans = src_dict["answer_idx"]
            if isinstance(ans, str):
                answer_id = ans
            elif isinstance(ans, list):
                answer_id = ",".join(map(str, ans))
            elif isinstance(ans, int):
                answer_id = chr(ord("A") + ans)
        elif "answer" in src_dict:
            ans = src_dict["answer"]
            if isinstance(ans, str):
                answer_id = ans
            elif isinstance(ans, int):
                answer_id = chr(ord("A") + ans)

    
        rationale = src_dict.get("rationale", "")

     
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
            }

        res_dict_ls.append(tmp)

    return res_dict_ls


def prepare_data(data_list: Sequence[dict]) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        data_list[_idx]['sample_id'] = _idx

        data_list[_idx]['pmc_input'] = (
            prompt_input.format_map(data_entry)
            if data_entry.get("input", "") != ""
            else prompt_no_input.format_map(data_entry)
        )
        data_list[_idx]['pmc_output'] = data_entry['output']     
        data_list[_idx]['rationale'] = data_entry['rationale']    
    return data_list


def inference(
    test_filepath,
    model,
    tokenizer,
    save_dir,
    is_with_rationale,
    batch_size,
    self_consistency_k: int = 1,
    sc_temperature: float = 0.7,
):

    from collections import Counter

    data_list = read_jsonl(test_filepath, is_with_rationale)
    data_list = prepare_data(data_list)

    filename = os.path.basename(test_filepath).split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_res.txt")

    answers = []

    for data_entry in tqdm(data_list):
        input_str = [data_entry['pmc_input']]  
        gold = data_entry['pmc_output']
        rationale = data_entry['rationale']

   
        if self_consistency_k <= 1:
        
            output_str = inference_on_one(input_str, model, tokenizer, temperature=0.0)

            if "### Response:" in output_str:
                response = output_str.split("### Response:", 1)[1].strip()
            else:
                response = output_str.strip()

            response = response.replace("\n", "")
            answers.append((response, gold, rationale))
            continue
      

    
        responses = []
        letters = []

        for _ in range(self_consistency_k):
      
            output_str = inference_on_one(input_str, model, tokenizer, temperature=sc_temperature)

            if "### Response:" in output_str:
                resp = output_str.split("### Response:", 1)[1].strip()
            else:
                resp = output_str.strip()

            resp = resp.replace("\n", "")
            responses.append(resp)

           
            letter = extract_option_letter(resp)
            letters.append(letter)


        valid_pairs = [
            (resp, letter) for resp, letter in zip(responses, letters) if letter != ""
        ]

        if valid_pairs:
         
            letter_counts = Counter([letter for _, letter in valid_pairs])
            majority_letter, _ = letter_counts.most_common(1)[0]

       
            chosen_response = None
            for resp, letter in valid_pairs:
                if letter == majority_letter:
                    chosen_response = resp
                    break
            if chosen_response is None:
              
                chosen_response = valid_pairs[0][0]

            final_response = chosen_response
        else:
     
            final_response = responses[0]

        answers.append((final_response, gold, rationale))
     

    with open(save_path, "w", encoding="utf-8") as fp:
        for response, target, rationale in answers:
            fp.write(f"{response}[SPLIT]{target}[SPLIT]{rationale}\n")


def validate():
    args = parse_args()
    filepaths = [
        os.path.join(args.data_path, filename)
        for filename in os.listdir(args.data_path)
        if filename.endswith('.jsonl')
    ]

    if args.is_lora == False:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        model = AutoPeftModel.from_pretrained(args.model_name_or_path)

    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=2048,
        use_fast=False,
        trust_remote_code=True
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    for idx, filepath in enumerate(filepaths):
        inference(
            filepath,
            model,
            tokenizer,
            args.save_dir,
            args.is_with_rationale,
            args.batch_size,
            self_consistency_k=args.self_consistency_k, 
        )
        print(idx, "\t", filepath, " has done!")


if __name__ == '__main__':
    validate()

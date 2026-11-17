
from typing import Dict, Any, List, Tuple

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

def build_messages(question: str, options_str: str, prompt_type: str) -> List[Dict[str, str]]:
    system_msg = USMLE_SYSTEM_COT if prompt_type == "cot" else USMLE_SYSTEM_SP
    user_msg = (question + " " + options_str).strip()  
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

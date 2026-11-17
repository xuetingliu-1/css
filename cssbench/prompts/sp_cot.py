from typing import Dict, Any, List, Tuple
LETTER_5 = ["A", "B", "C", "D", "E"]

def format_options_block(options: Dict[str, str]) -> str:
    lines = []
    for k in LETTER_5:
        if k in options:
            lines.append(f"{k}. {str(options[k]).strip()}")
    return "\n".join(lines)

def build_messages(question: str, options: Dict[str, str], mode: str, use_system: bool) -> List[Dict[str, str]]:
    opt_block = format_options_block(options)

    system_msg = (
        "You are an expert in clinical reasoning for USMLE-style multiple-choice questions.\n\n"
    )

    if mode == "sp":
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Given a question and a list of options, select the correct answer from the options directly."
        )
    elif mode == "cot":
        user = (
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_block}\n\n"
            "Given a question and a list of options,"
            "Let’s think step by step to answer the question. "
            "Select the correct answer from the options directly."
        )
    else:
        raise ValueError(f"Unknown mode={mode}")

    if use_system:
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user}]
    else:
        return [{"role": "user", "content": user}]

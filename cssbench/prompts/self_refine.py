
from typing import Dict, Any, List, Tuple

def format_options_medins(raw_options) -> str:
    """
    MedIns style: "A: ... \t B: ... \t C: ..."
    """
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

def sr_init_instruction() -> str:
    return "Given a question and a list of options, select the correct answer from the options directly."

def sr_feedback_instruction() -> str:
    return (
        "You are an expert clinical reviewer.\n"
        "Given the question, options, and the proposed answer, write actionable and specific feedback.\n"
        "If the proposed answer is clearly correct, write 'Stop: YES'. Otherwise write 'Stop: NO'.\n"
        "Output format:\n"
        "Feedback: <your feedback>\n"
        "Stop: YES/NO"
    )

def sr_refine_instruction() -> str:
    return (
        "Revise the answer using the feedback.\n"
        "In your final output, ONLY provide the letter of the best option (A, B, C, D, or E) "
        "without any additional text."
    )

def build_messages(system_inst: str, user_query: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_query},
    ]

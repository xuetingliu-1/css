from typing import Dict, Any, List, Tuple

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


from typing import Dict, Any, List, Tuple

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

def build_messages(system_inst: str, user_query: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_inst}, {"role": "user", "content": user_query}]

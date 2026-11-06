from __future__ import annotations

from .core import QAItem

ROUTER_SYS = ("You triage biomedical questions. A SIMPLE question can be answered from one or two passages "
              "with a single search. A COMPLEX question needs several facts, comparisons, causal chains or "
              "multiple entities, so it requires multi-step evidence gathering. Reply with exactly one word: "
              "SIMPLE or COMPLEX.")


def router_msgs(q: str) -> list:
    return [{"role": "system", "content": ROUTER_SYS}, {"role": "user", "content": f"Question: {q}"}]


def question_block(item: QAItem) -> str:
    if item.options:
        opts = "\n".join(f"{k}. {v}" for k, v in item.options.items())
        return f"{item.question}\nOptions:\n{opts}"
    return item.question


def decompose_msgs(item: QAItem, max_sub: int) -> list:
    sys = ("You decompose complex biomedical questions into minimal, self-contained sub-questions, each "
           f"answerable from the literature with one search. Produce 2 to {max_sub} non-overlapping "
           "sub-questions; resolve pronouns so each reads standalone. If the question cannot be split, "
           'return it unchanged. Output JSON only: {"subquestions": ["...", "..."]}')
    return [{"role": "system", "content": sys}, {"role": "user", "content": question_block(item)}]


def sufficiency_msgs(question: str, sub: str, passages: list) -> list:
    sys = ("You judge evidence sufficiency. Given a sub-question and numbered passages, decide whether the "
           "passages TOGETHER contain enough information to answer the sub-question. Be strict: topical "
           "overlap alone is not sufficient. Output JSON only: "
           '{"sufficient": true|false, "missing": "<short search phrase for the missing information, '
           'empty if sufficient>"}')
    ctx = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(no passages)"
    user = f"Original question: {question}\nSub-question: {sub}\n\nPassages:\n{ctx}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


GEN_SYS = (
    "You are a careful biomedical expert. Answer ONLY from the numbered documents provided. "
    "Cite the supporting documents inline as [Doc-N] (e.g. [Doc-2]); never cite a number that is not "
    "in the list. If the documents are insufficient, give your best-supported answer and say what is "
    "uncertain."
)

_FORMAT = {
    "pubmedqa": (
        "Answer the biomedical research question using the conclusion supported by the provided documents. "
        "Use the paper's reported findings and conclusion to make the yes/no/maybe decision. "
        "Do not choose maybe merely because the study has limitations, the result is not definitive, "
        "or further research is recommended. Choose maybe only when the provided evidence genuinely "
        "does not support either yes or no. Do not infer a stronger claim than the paper reports.\n"
        "Answer yes, no or maybe.\n"
        "Format:\n"
        "Answer: <yes|no|maybe>\n"
        "Explanation: <1-3 sentences with [Doc-N] citations>\n"
        "\n"
        "Before choosing yes, no, or maybe, identify the exact claim asked by the question and compare "
        "it with the study's reported conclusion. Do not treat an association, recommendation, observed "
        "difference, or statistical finding as automatically supporting the question's claim. Check "
        "whether it answers the same population, intervention or exposure, comparison, and outcome. "
        "Choose yes or no when the provided evidence supports or contradicts the claim, even if the "
        "study has limitations or recommends further research. Choose maybe only when the evidence "
        "genuinely leaves the claim unresolved or the provided documents do not support either direction. "
        "Do not infer a stronger conclusion than the study reports."
    ),
    "mcq": (
        "Choose the single best option.\n"
        "Format:\n"
        "Answer: <option letter>\n"
        "Explanation: <1-3 sentences with [Doc-N] citations>"
    ),
    "yesno": (
        "Answer yes or no.\n"
        "Format:\n"
        "Answer: <yes|no>\n"
        "Explanation: <1-3 sentences with [Doc-N] citations>"
    ),
    "factoid": (
        "Give a short factoid answer (an entity, number or short phrase).\n"
        "Format:\n"
        "Answer: <short answer>\n"
        "Explanation: <1-2 sentences with [Doc-N] citations>"
    ),
    "list": (
        "Give a list of entities.\n"
        "Format:\n"
        "Answer: <item 1>; <item 2>; ...\n"
        "Explanation: <1-2 sentences with [Doc-N] citations>"
    ),
    "summary": (
        "Write a concise, self-contained summary answer (3-5 sentences) with inline [Doc-N] citations.\n"
        "Format:\n"
        "Answer: <summary with citations>"
    ),
}

def gen_msgs(item: QAItem, passages: list, char_limit: int, retry: bool = False) -> list:
    ctx = "\n\n".join(f"[Doc-{i + 1}] {p[:char_limit]}" for i, p in enumerate(passages)) or "(no documents)"
    user = f"Documents:\n{ctx}\n\nQuestion: {question_block(item)}\n\n{_FORMAT[item.qtype]}"
    if retry:
        user += (f"\n\nIMPORTANT: your previous answer had no valid citations. Cite using only "
                 f"[Doc-1] to [Doc-{len(passages)}].")
    return [{"role": "system", "content": GEN_SYS}, {"role": "user", "content": user}]

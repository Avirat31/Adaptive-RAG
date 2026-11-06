"""LLM-driven components: complexity estimation, decomposition, evidence sufficiency."""
from __future__ import annotations

import re

from .config import Config
from .core import QAItem
from .llm import LLM
from .prompts import decompose_msgs, router_msgs, sufficiency_msgs
from .utils import extract_json

_COMPLEX_CUES = re.compile(
    r"\b(compare|compared|versus|vs\.?|difference|differences|relationship|mechanism|mechanisms|"
    r"why|how does|how do|both|respectively|in addition|as well as|which of the following|"
    r"impact|interplay|combination|pathway|pathways|risk factors|complications)\b", re.I)


def heuristic_complexity(q: str) -> float:
    """Cheap score in [0, 1]; higher = more likely to need multi-step retrieval."""
    words = len(q.split())
    s = 0.0
    s += min(words / 60.0, 0.4)
    s += 0.15 * min(len(_COMPLEX_CUES.findall(q)), 2)
    s += 0.10 * min(len(re.findall(r"\b(and|or)\b", q, re.I)), 3)
    s += 0.10 * max(q.count("?") - 1, 0)
    s += 0.05 * min(q.count(","), 3)
    return min(s, 1.0)


class ComplexityRouter:
    def __init__(self, cfg: Config, llm: LLM | None):
        self.cfg, self.llm = cfg, llm
        self.lo, self.hi = 0.25, 0.60

    def classify(self, questions: list) -> list:
        mode = self.cfg.router_mode
        scores = [heuristic_complexity(q) for q in questions]
        labels = [None] * len(questions)
        need_llm = []
        for i, s in enumerate(scores):
            if mode == "heuristic" or (mode == "hybrid" and (s <= self.lo or s >= self.hi)):
                labels[i] = "complex" if s >= (self.hi if mode == "hybrid" else 0.45) else "simple"
            else:
                need_llm.append(i)
        if need_llm and self.llm is not None:
            outs = self.llm.chat([router_msgs(questions[i]) for i in need_llm], max_tokens=4)
            for i, o in zip(need_llm, outs):
                labels[i] = "complex" if "COMPLEX" in o.upper() else "simple"
        return [l or "simple" for l in labels]


def decompose_batch(llm: LLM, items: list, cfg: Config) -> list:
    outs = llm.chat([decompose_msgs(it, cfg.max_subquestions) for it in items], max_tokens=300)
    res = []
    for it, o in zip(items, outs):
        subs = []
        data = extract_json(o)
        if isinstance(data, dict):
            subs = [s.strip() for s in data.get("subquestions", []) if isinstance(s, str) and s.strip()]
        elif isinstance(data, list):
            subs = [s.strip() for s in data if isinstance(s, str) and s.strip()]
        subs = list(dict.fromkeys(subs))[: cfg.max_subquestions]
        res.append(subs or [it.question])
    return res


def assess_batch(llm: LLM, jobs: list) -> list:
    """jobs: [(question, subquestion, [passage_text,...])] -> [(sufficient: bool, missing: str)]"""
    todo = [i for i, (_, _, ps) in enumerate(jobs) if ps]
    verdicts = [(False, "no relevant evidence retrieved")] * len(jobs)
    if todo:
        outs = llm.chat([sufficiency_msgs(*jobs[i]) for i in todo], max_tokens=80)
        for i, o in zip(todo, outs):
            d = extract_json(o)
            if isinstance(d, dict) and "sufficient" in d:
                suf = d["sufficient"]
                suf = suf if isinstance(suf, bool) else str(suf).lower() == "true"
                verdicts[i] = (suf, str(d.get("missing", "") or "").strip())
            else:                                   # unparsable: avoid wasteful loops
                verdicts[i] = (True, "")
    return verdicts

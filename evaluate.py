"""Metrics: PubMedQA (acc/macro-F1), BioASQ (per type), MIRAGE (acc per subset) + RAG diagnostics."""
from __future__ import annotations

import re
from collections import Counter, defaultdict


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())

# def _yn(pred: str, labels=("yes", "no", "maybe")) -> str:
#     w = re.findall(r"[a-z]+", pred.lower())
#     for x in w[:3]:
#         if x in labels:
#             return x
#     return "maybe" if "maybe" in labels else "no"

def _yn(pred: str, labels=("yes", "no", "maybe")) -> str:
    text = str(pred).strip().lower()
    words = re.findall(r"[a-z]+", text)

    # Prefer an exact answer or a clear leading label.
    if words and words[0] in labels:
        return words[0]

    # Handle forms such as "The answer is yes".
    for pattern in (
        r"\banswer\s*(?:is|:)?\s*(yes|no|maybe)\b",
        r"\b(final answer|conclusion)\s*(?:is|:)?\s*(yes|no|maybe)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.groups()[-1]

    # Fallback: find a label anywhere in the extracted answer.
    found = [word for word in words if word in labels]
    if len(set(found)) == 1:
        return found[0]

    # Ambiguous or unrecognized answer.
    return "maybe" if "maybe" in labels else "no"


def _letter(pred: str) -> str:
    m = re.search(r"\b([A-E])\b", pred.strip().upper())
    return m.group(1) if m else ""


def macro_f1(gold: list, pred: list) -> float:
    f1s = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
        f1s.append(2 * pr * rc / (pr + rc + 1e-9))
    return sum(f1s) / max(len(f1s), 1)


def _lcs(a: list, b: list) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, refs: list) -> float:
    p = _norm(pred).split()
    best = 0.0
    for r in refs:
        rt = _norm(r).split()
        if not p or not rt:
            continue
        l = _lcs(p, rt)
        pr, rc = l / len(p), l / len(rt)
        best = max(best, 2 * pr * rc / (pr + rc + 1e-9))
    return best


def _flat_syn(exact):
    """BioASQ exact answers -> list of synonym-sets."""
    if exact is None:
        return []
    if isinstance(exact, str):
        return [[exact]]
    return [[e] if isinstance(e, str) else list(e) for e in exact]


def evaluate(traces: list) -> dict:
    out: dict = {}
    by = defaultdict(list)
    for t in traces:
        by[t.item.qtype].append(t)

    if "pubmedqa" in by:
        g = [t.item.gold for t in by["pubmedqa"]]
        p = [_yn(t.answer) for t in by["pubmedqa"]]
        out["pubmedqa"] = {"accuracy": sum(a == b for a, b in zip(g, p)) / len(g), "macro_f1": macro_f1(g, p)}

    if "mcq" in by:
        per = defaultdict(list)
        for t in by["mcq"]:
            per[t.item.meta.get("source", "all")].append(_letter(t.answer) == t.item.gold)
        out["mirage"] = {k: sum(v) / len(v) for k, v in per.items()}
        allv = [x for v in per.values() for x in v]
        out["mirage"]["average"] = sum(allv) / len(allv)

    bio = {}
    if "yesno" in by:
        g = [str(t.item.gold["exact"]).lower() for t in by["yesno"]]
        p = [_yn(t.answer, ("yes", "no")) for t in by["yesno"]]
        bio["yesno"] = {"accuracy": sum(a == b for a, b in zip(g, p)) / len(g), "macro_f1": macro_f1(g, p)}
    if "factoid" in by:
        strict = lenient = 0
        for t in by["factoid"]:
            syn = {_norm(s) for grp in _flat_syn(t.item.gold["exact"]) for s in grp}
            pa = _norm(t.answer)
            strict += pa in syn
            lenient += any(s and (s in pa or pa in s) for s in syn if pa)
        n = len(by["factoid"])
        bio["factoid"] = {"strict_acc": strict / n, "lenient_acc": lenient / n}
    if "list" in by:
        f1s = []
        for t in by["list"]:
            gold = {_norm(s) for grp in _flat_syn(t.item.gold["exact"]) for s in grp[:1]}
            pred = {_norm(x) for x in re.split(r"[;\n]", t.answer) if x.strip()}
            tp = len(gold & pred)
            pr, rc = tp / (len(pred) + 1e-9), tp / (len(gold) + 1e-9)
            f1s.append(2 * pr * rc / (pr + rc + 1e-9))
        bio["list"] = {"f1": sum(f1s) / len(f1s)}
    if "summary" in by:
        r = [rouge_l(t.answer, t.item.gold["ideal"] if isinstance(t.item.gold["ideal"], list)
                     else [t.item.gold["ideal"] or ""]) for t in by["summary"]]
        bio["summary"] = {"rouge_l_f": sum(r) / len(r)}
    if bio:
        out["bioasq"] = bio

    n = len(traces)
    out["diagnostics"] = {
        "n": n,
        "route_distribution": dict(Counter(t.route for t in traces)),
        "stop_reasons": dict(Counter(t.state.stop_reason for t in traces)),
        "avg_passages": sum(len(t.final_pids) for t in traces) / n,
        "avg_iterations_complex": (sum(t.iterations for t in traces if t.route == "complex") /
                                   max(1, sum(t.route == "complex" for t in traces))),
        "citation_valid_rate": sum(t.citation_ok for t in traces) / n,
        "citation_invalid_rate": sum(bool(t.invalid_cited) for t in traces) / n,
    }
    return out

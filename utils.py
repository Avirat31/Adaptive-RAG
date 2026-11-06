from __future__ import annotations

import json
import re

STOP = frozenset(
    "a an the of in on at to for with and or but is are was were be been being by from as that this these "
    "those it its which who whom what when where why how do does did can could should would may might will "
    "than then there their they them he she his her we you your not no yes also into over under between "
    "among about after before during such any each other more most some using use used versus".split())
_TERM_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
_GROUP_RE = re.compile(r"\[([^\[\]]*Doc-[^\[\]]*)\]")


def content_terms(text: str) -> set:
    return {t for t in _TERM_RE.findall(text.lower()) if t not in STOP}


def extract_json(text: str):
    """Best-effort extraction of the first JSON object/array from an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    spans = []
    for o, c in (("{", "}"), ("[", "]")):
        s, e = text.find(o), text.rfind(c)
        if s != -1 and e > s:
            spans.append((s, e))
    for s, e in sorted(spans):
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            continue
    return None


def find_citations(text: str) -> list:
    out = []
    for m in _GROUP_RE.finditer(text):
        out.extend(int(x) for x in re.findall(r"\d+", m.group(1)))
    return out


def validate_citations(text: str, n_passages: int):
    """Keep only [Doc-N] markers with 1 <= N <= n_passages. Returns (cleaned, valid, invalid)."""
    valid, invalid = set(), set()

    def repl(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        ok = [x for x in nums if 1 <= x <= n_passages]
        valid.update(ok)
        invalid.update(x for x in nums if x not in ok)
        return "[" + ", ".join(f"Doc-{x}" for x in ok) + "]" if ok else ""

    cleaned = _GROUP_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
    return cleaned, sorted(valid), sorted(invalid)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

"""Dataset + corpus loaders for PubMedQA, BioASQ and MIRAGE."""
from __future__ import annotations

import json
import os

from .core import Passage, QAItem


# --------------------------------------------------------------------------- corpus helpers
def chunk_text(text: str, words: int = 200, overlap: int = 40) -> list:
    w = text.split()
    if len(w) <= words:
        return [text]
    out, step = [], words - overlap
    for s in range(0, len(w), step):
        out.append(" ".join(w[s:s + words]))
        if s + words >= len(w):
            break
    return out


def load_corpus_jsonl(path: str, chunk_words: int = 200) -> list:
    """jsonl rows: {"id", "title", "contents"|"content"|"text"}  (MedCorp / PubMed-style)."""
    passages = []
    with open(path) as f:
        for ln, line in enumerate(f):
            r = json.loads(line)
            body = r.get("contents") or r.get("content") or r.get("text") or ""
            for c, ch in enumerate(chunk_text(body, chunk_words)):
                passages.append(Passage(len(passages), f"{r.get('id', ln)}#{c}", r.get("title", ""), ch))
    return passages


class _Pool:
    """Builds a de-duplicated corpus from benchmark-provided contexts."""

    def __init__(self):
        self.passages, self._seen = [], set()

    def add(self, doc_id, title, text):
        text = text.strip()
        if text and text not in self._seen:
            self._seen.add(text)
            self.passages.append(Passage(len(self.passages), str(doc_id), title, text))


# --------------------------------------------------------------------------- PubMedQA
def load_pubmedqa(path: str = "datasets/pubmedqa/ori_pqal.json",
                  split_name: str = "pqa_labeled",
                  limit: int | None = None):
    """Load the local PubMedQA ori_pqal.json file.

    Each record contains QUESTION, CONTEXTS, final_decision and LONG_ANSWER.
    The contexts form the default retrieval corpus.
    """
    with open(path) as f:
        data = json.load(f)

    items, pool = [], _Pool()
    for pubid, r in data.items():
        question = r.get("QUESTION", "").strip()
        contexts = r.get("CONTEXTS", [])
        gold = str(r.get("final_decision", "")).strip().lower()

        if not question or gold not in {"yes", "no", "maybe"}:
            continue

        for i, context in enumerate(contexts):
            # A context can contain multiple sentences; chunk_text keeps
            # longer contexts within the configured retrieval chunk size.
            for j, chunk in enumerate(chunk_text(context)):
                pool.add(
                    f"{pubid}_{i}_{j}",
                    f"PubMed {pubid}",
                    chunk,
                )

        items.append(QAItem(
            str(pubid),
            question,
            "pubmedqa",
            "pubmedqa",
            gold=gold,
            meta={"long_answer": r.get("LONG_ANSWER", "")},
        ))

        if limit is not None and len(items) >= limit:
            break

    return items, pool.passages


# --------------------------------------------------------------------------- BioASQ
def load_bioasq(path: str, limit: int | None = None):
    """BioASQ training/test JSON ({"questions": [...]}). Snippets are pooled as the default corpus."""
    data = json.load(open(path))["questions"]
    items, pool = [], _Pool()
    for r in data:
        for s in r.get("snippets", []):
            pool.add(s.get("document", "snippet"), "", s["text"])
        items.append(QAItem(r["id"], r["body"], "bioasq", r["type"],
                            gold={"exact": r.get("exact_answer"), "ideal": r.get("ideal_answer")}))
    return (items[:limit] if limit else items), pool.passages


# --------------------------------------------------------------------------- MIRAGE
def load_mirage(path: str, subsets: list | None = None, limit: int | None = None):
    """MIRAGE benchmark.json: {subset: {qid: {question, options{A..}, answer}}}.
    A retrieval corpus (e.g. MedCorp chunks) must be supplied separately."""
    data = json.load(open(path))
    items = []
    for sub, qs in data.items():
        if subsets and sub not in subsets:
            continue
        for qid, r in qs.items():
            items.append(QAItem(f"{sub}/{qid}", r["question"], "mirage", "mcq",
                                options=r["options"], gold=r["answer"], meta={"source": sub}))
    return (items[:limit] if limit else items), None


def load_dataset_and_corpus(args, cfg):
    if args.dataset == "pubmedqa":
        items, passages = load_pubmedqa(limit=args.limit)
    elif args.dataset == "bioasq":
        items, passages = load_bioasq(args.data_path, limit=args.limit)
    elif args.dataset == "mirage":
        items, passages = load_mirage(args.data_path, args.mirage_subsets, args.limit)
    else:
        raise ValueError(args.dataset)
    if args.corpus_path:
        passages = load_corpus_jsonl(args.corpus_path)
    if not passages:
        raise SystemExit("MIRAGE needs --corpus_path (e.g. MedCorp chunks as jsonl).")
    return items, passages

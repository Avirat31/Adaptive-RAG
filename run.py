"""AdaptiveBioRAG entry point.

Example (4 GPUs):
  python run.py --dataset pubmedqa --limit 200 --num_gpus 4 --tp_size 4
  python run.py --dataset bioasq --data_path BioASQ-training.json --corpus_path pubmed_chunks.jsonl
  python run.py --dataset mirage --data_path benchmark.json --corpus_path medcorp.jsonl --mode complex_only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time


def parse():
    from adaptive_biorag.config import Config
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["pubmedqa", "bioasq", "mirage"])
    p.add_argument("--data_path")
    p.add_argument("--corpus_path", help="jsonl corpus; default = pool of benchmark contexts")
    p.add_argument("--mirage_subsets", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--cache_dir", default=".cache")
    d = Config()
    for f in dataclasses.fields(Config):
        t = f.type
        v = getattr(d, f.name)
        if isinstance(v, bool):
            p.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{f.name}", type=type(v) if v is not None else int, default=v)
    return p.parse_args()


def main():
    args = parse()
    from adaptive_biorag.config import Config
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})

    from adaptive_biorag.data import load_dataset_and_corpus
    from adaptive_biorag.evaluate import evaluate
    from adaptive_biorag.gpu_utils import get_gpu_ids
    from adaptive_biorag.llm import LLM
    from adaptive_biorag.pipeline import AdaptiveBioRAG
    from adaptive_biorag.retrieval import (BM25Retriever, HashDenseIndex, HybridRetriever,
                                           PassageStore, QueryEncoder, embed_corpus)
    from adaptive_biorag.utils import chunks

    print(f"GPUs: {get_gpu_ids(cfg.num_gpus)}")
    items, passages = load_dataset_and_corpus(args, cfg)
    store = PassageStore(passages)
    print(f"{len(items)} questions, {len(store)} passages")

    # Order matters for GPU memory: (1) corpus embeddings on all GPUs, (2) LLM (vLLM claims
    # gpu_mem_util of every GPU), (3) dense-index shards + query encoder in the remaining headroom.
    store.vecs = embed_corpus(store, cfg, args.cache_dir)
    llm = LLM(cfg)
    qenc = QueryEncoder(cfg)
    dense = HashDenseIndex(store.vecs, cfg)
    bm25 = BM25Retriever(store)
    pipe = AdaptiveBioRAG(cfg, store, HybridRetriever(cfg, bm25, dense, qenc), llm)

    traces, t0 = [], time.time()
    for i, batch in enumerate(chunks(items, cfg.batch_size)):
        traces += pipe.run(batch)
        print(f"batch {i + 1}: {len(traces)}/{len(items)}  ({time.time() - t0:.0f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.dataset}_{cfg.mode}"
    with open(os.path.join(args.output_dir, f"{tag}_predictions.jsonl"), "w") as f:
        for t in traces:
            f.write(json.dumps({
                "qid": t.item.qid, "question": t.item.question, "gold": t.item.gold, "route": t.route,
                "subquestions": t.subquestions, "evidence_state": t.state.to_dict(),
                "iterations": t.iterations,
                "passages": [{"cite": f"Doc-{i + 1}", "doc_id": store.passages[p].doc_id}
                             for i, p in enumerate(t.final_pids)],
                "answer": t.answer, "explanation": t.explanation, "raw": t.raw_answer,
                "valid_citations": t.cited, "invalid_citations": t.invalid_cited,
                "citation_ok": t.citation_ok, "gen_retried": t.gen_retried}, default=str) + "\n")
    metrics = evaluate(traces)
    metrics["config"] = cfg.to_dict()
    json.dump(metrics, open(os.path.join(args.output_dir, f"{tag}_metrics.json"), "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":      # required: the multi-GPU embedding pool spawns worker processes
    main()

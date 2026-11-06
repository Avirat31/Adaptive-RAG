# AdaptiveBioRAG

Adaptive, coverage-aware RAG for multi-step biomedical QA (PubMedQA, BioASQ, MIRAGE).

## Pipeline -> module map
| Step | Module |
|---|---|
| Complexity estimator & router (heuristic + LLM) | `reasoning.py` (`ComplexityRouter`), `pipeline._route` |
| Question decomposition, evidence sufficiency | `reasoning.py` |
| BM25 + hashing dense retrieval + RRF | `retrieval.py` |
| Coverage-aware selection, final evidence assembly | `selection.py` |
| Evidence state (`covered`, `unresolved`, `stop_reason`), iterative loop | `core.py`, `pipeline._complex_path` |
| Grounded generation + citation validation | `prompts.py`, `pipeline._generate/_finalise`, `utils.validate_citations` |
| Datasets / metrics | `data.py`, `evaluate.py` |

## Multi-GPU usage
- **Corpus embedding**: `sentence-transformers` multi-process pool, one worker per GPU, cached to `.cache/`.
- **Dense retrieval**: index sharded across GPUs. Stage 1 scans sign-random-projection codes (Hamming via one fp16 matmul), stage 2 re-scores the top candidates exactly. `--use_hashing false` gives exact search.
- **LLM**: vLLM tensor parallel over `--tp_size` GPUs (must divide the model's attention heads). `--llm_backend hf` uses `device_map=auto` instead.
- Questions run in lock-step batches (`--batch_size`), so routing, decomposition, sufficiency checks and generation are each one batched call.
- GPU memory order: embeddings -> vLLM (`--gpu_mem_util 0.80`) -> index shards + query encoder in the remaining headroom. Lower `--gpu_mem_util` if the shards do not fit.
- BM25 (`bm25s`) runs on CPU.

## Run
```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # CPU logic check with mocks
python run.py --dataset pubmedqa --limit 200 --num_gpus 4 --tp_size 4
python run.py --dataset bioasq --data_path BioASQ-training.json
python run.py --dataset mirage --data_path benchmark.json --corpus_path medcorp.jsonl
# ablations for MIRAGE-style RAG configuration comparison:
python run.py ... --mode simple_only    # / complex_only / adaptive
```
Outputs: `outputs/<dataset>_<mode>_predictions.jsonl` (route, sub-questions, evidence state, cited passages) and `_metrics.json`.

## Notes
- PubMedQA / BioASQ default corpora pool the benchmark's own contexts/snippets. This is an easy setting; pass `--corpus_path` (jsonl with `id,title,contents`) for a realistic PubMed corpus. MIRAGE requires `--corpus_path` (e.g. MedCorp).
- Everything is untested against real models/GPUs in my environment (no network); only `tests/smoke_test.py` was run. Expect to tune `min_utility`, `dup_threshold`, and the router thresholds per dataset.
- BioASQ factoid/list scoring is a simplified lenient approximation, not the official BioASQ evaluator.

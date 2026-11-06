from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # ---- models -----------------------------------------------------------
    embed_model: str = "BAAI/bge-large-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_seq_length: int = 512
    embed_batch_size: int = 256
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_backend: str = "vllm"            # "vllm" (tensor parallel) | "hf" (device_map=auto)
    num_gpus: int | None = None          # None -> use every visible GPU
    tp_size: int | None = None           # vLLM tensor-parallel size; None -> num_gpus
    gpu_mem_util: float = 0.80           # leave ~20% free for dense-index shards + query encoder
    max_model_len: int = 8192
    dtype: str = "bfloat16"

    # ---- retrieval ----------------------------------------------------------
    bm25_k: int = 100
    dense_k: int = 100
    rrf_k: int = 60                      # RRF constant
    fused_k: int = 30                    # candidates passed to selection
    use_hashing: bool = True             # sign-random-projection codes + exact re-scoring
    hash_bits: int = 1024
    hash_seed: int = 13
    rescore_mult: int = 8                # re-score top (k * mult) Hamming candidates exactly

    # ---- selection ----------------------------------------------------------
    alpha_rel: float = 1.0               # relevance weight
    beta_gain: float = 0.6               # new query-term coverage weight
    gamma_red: float = 0.8               # redundancy penalty
    dup_threshold: float = 0.93          # cosine above which two passages are duplicates
    min_utility: float = 0.20            # stop adding passages below this marginal utility
    simple_budget: int = 5
    per_sub_budget: int = 3
    max_context: int = 8                 # passages given to the generator

    # ---- routing / iteration -----------------------------------------------
    mode: str = "adaptive"               # adaptive | simple_only | complex_only
    router_mode: str = "hybrid"          # heuristic | llm | hybrid
    max_subquestions: int = 4
    max_iterations: int = 3

    # ---- generation ---------------------------------------------------------
    max_new_tokens: int = 384
    passage_char_limit: int = 1200
    batch_size: int = 64                 # questions processed in lock-step per batch

    def to_dict(self):
        return asdict(self)

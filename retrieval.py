"""Hybrid retrieval: BM25 (sparse) + hashing dense retrieval (multi-GPU sharded) fused by RRF."""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .config import Config
from .core import Passage
from .gpu_utils import get_gpu_ids


# --------------------------------------------------------------------------- store
class PassageStore:
    def __init__(self, passages: list):
        self.passages: list = passages
        self.vecs: np.ndarray | None = None      # (N, D) fp16, L2-normalised
        self._terms: dict = {}

    def __len__(self):
        return len(self.passages)

    def text(self, pid: int) -> str:
        p = self.passages[pid]
        return f"{p.title}. {p.text}" if p.title else p.text

    def get_vectors(self, pids) -> np.ndarray:
        return np.asarray(self.vecs[list(pids)], dtype=np.float32)

    def terms(self, pid: int) -> set:
        from .utils import content_terms
        if pid not in self._terms:
            self._terms[pid] = content_terms(self.text(pid))
        return self._terms[pid]


# --------------------------------------------------------------------------- embeddings
def embed_corpus(store: PassageStore, cfg: Config, cache_dir: str) -> np.ndarray:
    """Encode the corpus on ALL visible GPUs (one worker process per GPU) and cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    sig = hashlib.md5((cfg.embed_model + str(len(store)) + store.text(0)[:200] +
                       store.text(len(store) - 1)[-200:]).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"emb_{sig}.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")

    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.embed_model, device="cpu", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = cfg.max_seq_length
    gpus = get_gpu_ids(cfg.num_gpus)
    devices = [f"cuda:{i}" for i in gpus] or ["cpu"]
    texts = [store.text(i) for i in range(len(store))]
    pool = model.start_multi_process_pool(devices)
    try:
        emb = model.encode_multi_process(texts, pool, batch_size=cfg.embed_batch_size,
                                         chunk_size=max(1024, cfg.embed_batch_size * 8),
                                         normalize_embeddings=True)
    finally:
        SentenceTransformer.stop_multi_process_pool(pool)
    emb = np.asarray(emb, dtype=np.float16)
    np.save(path, emb)
    return np.load(path, mmap_mode="r")


class QueryEncoder:
    def __init__(self, cfg: Config):
        import torch
        from sentence_transformers import SentenceTransformer
        gpus = get_gpu_ids(cfg.num_gpus)
        dev = f"cuda:{gpus[0]}" if gpus else "cpu"
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embed_model, device=dev, model_kwargs={"torch_dtype": torch.float16})
        self.model.max_seq_length = cfg.max_seq_length

    def encode(self, queries: list) -> np.ndarray:
        q = [self.cfg.query_prefix + x for x in queries]
        return self.model.encode(q, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


# --------------------------------------------------------------------------- sparse
class BM25Retriever:
    def __init__(self, store: PassageStore, k1: float = 1.2, b: float = 0.75):
        import bm25s
        self.bm25s = bm25s
        try:
            import Stemmer
            self.stemmer = Stemmer.Stemmer("english")
        except ImportError:
            self.stemmer = None
        self.n = len(store)
        self.ret = bm25s.BM25(k1=k1, b=b)
        self.ret.index(self._tok([store.text(i) for i in range(self.n)]), show_progress=False)

    def _tok(self, texts):
        toks = self.bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer,
                                   return_ids=False, show_progress=False)
        return [t if t else ["__empty__"] for t in toks]

    def search(self, queries: list, k: int):
        k = min(k, self.n)
        docs, scores = self.ret.retrieve(self._tok(queries), k=k, show_progress=False)
        return [[(int(d), float(s)) for d, s in zip(dr, sr) if s > 0] for dr, sr in zip(docs, scores)]


# --------------------------------------------------------------------------- dense (hashing)
class HashDenseIndex:
    """Dense retrieval sharded over all GPUs.

    Stage 1 (hashing): sign-random-projection codes (+/-1, fp16); the dot product of two codes is an
    affine function of their Hamming distance, so a single GPU matmul scans the whole shard.
    Stage 2: the top (k * rescore_mult) Hamming candidates are re-scored with the exact fp16 embeddings.
    With use_hashing=False it falls back to an exact sharded matmul."""

    def __init__(self, emb: np.ndarray, cfg: Config):
        import torch
        self.torch = torch
        self.cfg = cfg
        gpus = get_gpu_ids(cfg.num_gpus)
        devs = [torch.device(f"cuda:{g}") for g in gpus] or [torch.device("cpu")]
        n, d = emb.shape
        bounds = np.linspace(0, n, len(devs) + 1).astype(int)
        planes = torch.randn(d, cfg.hash_bits, generator=torch.Generator().manual_seed(cfg.hash_seed))
        self.shards = []
        dt = torch.float16 if devs[0].type == "cuda" else torch.float32
        for dev, lo, hi in zip(devs, bounds[:-1], bounds[1:]):
            e = torch.from_numpy(np.ascontiguousarray(emb[lo:hi])).to(dev, dt)
            sh = {"dev": dev, "lo": int(lo), "emb": e, "codes": None, "planes": None}
            if cfg.use_hashing:
                pl = planes.to(dev, dt)
                codes = torch.empty(hi - lo, cfg.hash_bits, dtype=dt, device=dev)
                for s in range(0, hi - lo, 65536):
                    codes[s:s + 65536] = (e[s:s + 65536] @ pl > 0).to(dt) * 2 - 1
                sh["codes"], sh["planes"] = codes, pl
            self.shards.append(sh)
        self.dt = dt

    def search(self, q: np.ndarray, k: int, chunk: int = 128):
        s_all, i_all = [], []
        for st in range(0, len(q), chunk):
            s, i = self._search(q[st:st + chunk], k)
            s_all.append(s)
            i_all.append(i)
        return np.concatenate(s_all), np.concatenate(i_all)

    def _search(self, q: np.ndarray, k: int):
        torch = self.torch
        pend = []
        with torch.no_grad():
            for sh in self.shards:                       # launch on every GPU first (async)
                qt = torch.from_numpy(q).to(sh["dev"], self.dt)
                n = sh["emb"].shape[0]
                kk = min(k, n)
                if sh["codes"] is not None:
                    qc = (qt @ sh["planes"] > 0).to(self.dt) * 2 - 1
                    approx = qc @ sh["codes"].T
                    c = min(kk * self.cfg.rescore_mult, n)
                    cidx = approx.topk(c, dim=1).indices
                    exact = torch.einsum("bd,bcd->bc", qt, sh["emb"][cidx])
                    sc, pos = exact.topk(kk, dim=1)
                    idx = cidx.gather(1, pos)
                else:
                    sc, idx = (qt @ sh["emb"].T).topk(kk, dim=1)
                pend.append((sc, idx + sh["lo"]))
            scores = torch.cat([p[0].float().cpu() for p in pend], 1)
            ids = torch.cat([p[1].cpu() for p in pend], 1)
            top, pos = scores.topk(min(k, scores.shape[1]), dim=1)
            return top.numpy(), ids.gather(1, pos).numpy()


# --------------------------------------------------------------------------- fusion
def rrf_fuse(rank_lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion. rank_lists: list of [pid,...] ordered best-first."""
    score: dict = {}
    for lst in rank_lists:
        for r, pid in enumerate(lst, start=1):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, cfg: Config, bm25: BM25Retriever, dense: HashDenseIndex, qenc: QueryEncoder):
        self.cfg, self.bm25, self.dense, self.qenc = cfg, bm25, dense, qenc

    def encode(self, queries: list) -> np.ndarray:
        return self.qenc.encode(queries)

    def search(self, queries: list, qvecs: np.ndarray, exclude: list | None = None) -> list:
        """Returns, per query, [(pid, rrf_score)] best-first (top fused_k, minus excluded pids)."""
        sparse = self.bm25.search(queries, self.cfg.bm25_k)
        _, dense_ids = self.dense.search(qvecs, self.cfg.dense_k)
        out = []
        for i in range(len(queries)):
            fused = rrf_fuse([[p for p, _ in sparse[i]], [int(x) for x in dense_ids[i]]], self.cfg.rrf_k)
            if exclude is not None:
                fused = [x for x in fused if x[0] not in exclude[i]]
            out.append(fused[: self.cfg.fused_k])
        return out

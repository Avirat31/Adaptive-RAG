"""Coverage-aware passage selection (marginal utility + redundancy filtering)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .retrieval import PassageStore
from .utils import content_terms


def select_passages(cands: list, fused: dict, qvec: np.ndarray, query: str, store: PassageStore,
                    cfg: Config, budget: int) -> list:
    """Greedy marginal-utility selection.

    U(p | S) = alpha * rel(p) + beta * newly_covered_query_terms(p | S) - gamma * max_sim(p, S)
    rel(p) blends dense cosine with the (min-max normalised) RRF score. Near-duplicates
    (cos >= dup_threshold) are dropped; selection stops at `budget` or when U < min_utility."""
    if not cands:
        return []
    V = store.get_vectors(cands)
    cos = V @ qvec
    rrf = np.array([fused.get(p, 0.0) for p in cands], dtype=np.float32)
    rrf = (rrf - rrf.min()) / (rrf.max() - rrf.min() + 1e-9)
    cos_n = (cos - cos.min()) / (cos.max() - cos.min() + 1e-9)
    rel = 0.5 * cos_n + 0.5 * rrf
    sim = V @ V.T
    q_terms = content_terms(query)
    covered: set = set()
    chosen: list = []
    remaining = list(range(len(cands)))
    while remaining and len(chosen) < budget:
        best, best_u = None, -1e9
        for i in remaining:
            red = max((sim[i, j] for j in chosen), default=0.0)
            if red >= cfg.dup_threshold:
                continue
            new_terms = (store.terms(cands[i]) & q_terms) - covered
            gain = len(new_terms) / max(1, len(q_terms))
            u = cfg.alpha_rel * rel[i] + cfg.beta_gain * gain - cfg.gamma_red * max(red, 0.0)
            if u > best_u:
                best, best_u = i, u
        if best is None or (chosen and best_u < cfg.min_utility):
            break
        chosen.append(best)
        covered |= store.terms(cands[best]) & q_terms
        remaining.remove(best)
    return [cands[i] for i in chosen]


def assemble_evidence(sub_evidence: dict, question: str, qvec: np.ndarray, store: PassageStore,
                      cfg: Config) -> list:
    """Final evidence assembly: union over sub-questions, global de-duplication, and a cap of
    `max_context` passages that still keeps the best passage of every sub-question."""
    pids: list = []
    for lst in sub_evidence.values():
        for p in lst:
            if p not in pids:
                pids.append(p)
    if not pids:
        return []
    V = store.get_vectors(pids)
    rel = V @ qvec
    order = list(np.argsort(-rel))
    kept: list = []
    for i in order:                                       # global redundancy filter
        if all(float(V[i] @ V[j]) < cfg.dup_threshold for j in kept):
            kept.append(i)
    kept_pids = [pids[i] for i in kept]
    forced: list = []
    for lst in sub_evidence.values():                     # keep each sub-question's top passage
        for p in lst:
            if p in kept_pids:
                forced.append(p)
                break
    final = list(dict.fromkeys(forced))
    for p in kept_pids:
        if len(final) >= cfg.max_context:
            break
        if p not in final:
            final.append(p)
    rank = {p: r for r, p in enumerate(kept_pids)}
    return sorted(final[: cfg.max_context], key=lambda p: rank[p])

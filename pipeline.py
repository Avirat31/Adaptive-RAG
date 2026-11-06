"""AdaptiveBioRAG pipeline. Questions are processed in lock-step batches so every LLM call and
every retrieval call is batched, which is what keeps all GPUs busy."""
from __future__ import annotations

import re
from collections import defaultdict

from .config import Config
from .core import EvidenceState, QAItem, Trace
from .llm import LLM
from .prompts import gen_msgs
from .reasoning import ComplexityRouter, assess_batch, decompose_batch
from .retrieval import HybridRetriever, PassageStore
from .selection import assemble_evidence, select_passages
from .utils import validate_citations


class AdaptiveBioRAG:
    def __init__(self, cfg: Config, store: PassageStore, retriever: HybridRetriever, llm: LLM):
        self.cfg, self.store, self.ret, self.llm = cfg, store, retriever, llm
        self.router = ComplexityRouter(cfg, llm)

    # ------------------------------------------------------------------ public
    def run(self, items: list) -> list:
        traces = [Trace(item=it) for it in items]
        self._route(traces)
        self._simple_path([t for t in traces if t.route == "simple"])
        self._complex_path([t for t in traces if t.route == "complex"])
        self._generate(traces)
        return traces

    # ------------------------------------------------------------------ routing
    def _route(self, traces):
        if self.cfg.mode == "simple_only":
            labels = ["simple"] * len(traces)
        elif self.cfg.mode == "complex_only":
            labels = ["complex"] * len(traces)
        else:
            labels = self.router.classify([t.item.question for t in traces])
        for t, l in zip(traces, labels):
            t.route = l

    # ------------------------------------------------------------------ simple path
    def _simple_path(self, ts):
        if not ts:
            return
        qs = [t.item.question for t in ts]
        qv = self.ret.encode(qs)
        hits = self.ret.search(qs, qv)
        for t, q, v, h in zip(ts, qs, qv, hits):
            fused = dict(h)
            t.final_pids = select_passages([p for p, _ in h], fused, v, q, self.store, self.cfg,
                                           self.cfg.simple_budget)
            t.state = EvidenceState(covered=[q] if t.final_pids else [], unresolved=[],
                                    stop_reason="single_step")

    # ------------------------------------------------------------------ complex path
    def _complex_path(self, ts):
        if not ts:
            return
        cfg = self.cfg
        subs = decompose_batch(self.llm, [t.item for t in ts], cfg)
        for t, s in zip(ts, subs):
            t.subquestions = s
            t.state = EvidenceState(covered=[], unresolved=list(s), stop_reason=None)   # state init
            t.sub_evidence = {x: [] for x in s}

        for it in range(cfg.max_iterations):
            jobs = [(t, s) for t in ts if t.state.stop_reason is None for s in t.state.unresolved]
            if not jobs:
                break
            queries = [(s + " " + t.missing.get(s, "")).strip() for t, s in jobs]
            qv = self.ret.encode(queries)
            hits = self.ret.search(queries, qv, exclude=[set(t.sub_evidence[s]) for t, s in jobs])
            gained = defaultdict(int)
            for (t, s), q, v, h in zip(jobs, queries, qv, hits):
                sel = select_passages([p for p, _ in h], dict(h), v, q, self.store, cfg, cfg.per_sub_budget)
                new = [p for p in sel if p not in t.sub_evidence[s]]
                t.sub_evidence[s].extend(new)
                gained[id(t)] += len(new)

            verdicts = assess_batch(self.llm, [
                (t.item.question, s, [self.store.text(p)[: cfg.passage_char_limit] for p in t.sub_evidence[s]])
                for t, s in jobs])
            for (t, s), (ok, missing) in zip(jobs, verdicts):
                if ok:
                    t.state.unresolved.remove(s)
                    t.state.covered.append(s)
                    t.missing.pop(s, None)
                else:
                    t.missing[s] = missing

            active = {id(t): t for t, _ in jobs}
            for t in active.values():                                 # stop criteria
                t.iterations += 1
                if not t.state.unresolved:
                    t.state.stop_reason = "all_covered"
                elif gained[id(t)] == 0:
                    t.state.stop_reason = "no_new_evidence"
                elif it == cfg.max_iterations - 1:
                    t.state.stop_reason = "max_iterations"
        for t in ts:
            if t.state.stop_reason is None:
                t.state.stop_reason = "max_iterations"

        qv = self.ret.encode([t.item.question for t in ts])                   # final evidence assembly
        for t, v in zip(ts, qv):
            t.final_pids = assemble_evidence(t.sub_evidence, t.item.question, v, self.store, cfg)

    # ------------------------------------------------------------------ generation + citations
    def _generate(self, traces):
        cfg = self.cfg
        ptxt = {id(t): [self.store.text(p) for p in t.final_pids] for t in traces}
        outs = self.llm.chat([gen_msgs(t.item, ptxt[id(t)], cfg.passage_char_limit) for t in traces],
                             max_tokens=cfg.max_new_tokens)
        for t, o in zip(traces, outs):
            self._finalise(t, o)
        retry = [t for t in traces if t.final_pids and not t.cited]
        if retry:                                                   # one repair attempt, batched
            outs = self.llm.chat([gen_msgs(t.item, ptxt[id(t)], cfg.passage_char_limit, retry=True)
                                  for t in retry], max_tokens=cfg.max_new_tokens)
            for t, o in zip(retry, outs):
                t.gen_retried = True
                self._finalise(t, o)

    def _finalise(self, t: Trace, raw: str):
        t.raw_answer = raw
        cleaned, valid, invalid = validate_citations(raw, len(t.final_pids))   # citation validation
        t.cited, t.invalid_cited, t.citation_ok = valid, invalid, bool(valid) and not invalid
        m = re.search(r"Answer:\s*(.*?)(?:\n\s*Explanation:|\Z)", cleaned, re.S | re.I)
        t.answer = (m.group(1) if m else cleaned).strip()
        e = re.search(r"Explanation:\s*(.*)", cleaned, re.S | re.I)
        t.explanation = e.group(1).strip() if e else ""

"""CPU-only logic test with a mock LLM / retriever (no GPU, no model downloads)."""
import re, zlib
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_biorag.config import Config
from adaptive_biorag.core import Passage, QAItem
from adaptive_biorag.pipeline import AdaptiveBioRAG
from adaptive_biorag.retrieval import PassageStore, rrf_fuse
from adaptive_biorag.evaluate import evaluate
from adaptive_biorag.utils import validate_citations, extract_json

DIM = 64
def embed(text):
    v = np.zeros(DIM, np.float32)
    for w in re.findall(r"[a-z]+", text.lower()):
        v[zlib.crc32(w.encode()) % DIM] += 1
    return v / (np.linalg.norm(v) + 1e-9)

texts = ["metformin lowers blood glucose in type 2 diabetes", "metformin lowers blood glucose in type 2 diabetes patients",
         "statins reduce LDL cholesterol and cardiovascular risk", "aspirin inhibits platelet aggregation",
         "BRCA1 mutations raise breast cancer risk", "insulin resistance drives type 2 diabetes"]
store = PassageStore([Passage(i, str(i), "", t) for i, t in enumerate(texts)])
store.vecs = np.stack([embed(t) for t in texts]).astype(np.float16)

class FakeRet:
    def encode(self, qs): return np.stack([embed(q) for q in qs])
    def search(self, qs, qv, exclude=None):
        out = []
        for i, v in enumerate(qv):
            sc = store.vecs.astype(np.float32) @ v
            r = [(int(p), 1 / (60 + k + 1)) for k, p in enumerate(np.argsort(-sc)) if not exclude or p not in exclude[i]]
            out.append(r[:5])
        return out

class FakeLLM:
    def chat(self, convs, max_tokens=0, temperature=0.0):
        outs = []
        for c in convs:
            sys_, user = c[0]["content"], c[1]["content"]
            if "triage" in sys_: outs.append("COMPLEX" if "and" in user else "SIMPLE")
            elif "decompose" in sys_: outs.append('{"subquestions": ["What does metformin do?", "What drives type 2 diabetes?"]}')
            elif "sufficiency" in sys_: outs.append('{"sufficient": true, "missing": ""}')
            else: outs.append("Answer: yes\nExplanation: Supported [Doc-1] and bogus [Doc-99].")
        return outs

cfg = Config(mode="adaptive", router_mode="hybrid")
pipe = AdaptiveBioRAG(cfg, store, FakeRet(), FakeLLM())
items = [QAItem("1", "Does metformin lower glucose?", "pubmedqa", "pubmedqa", gold="yes"),
         QAItem("2", "How does metformin work and what drives type 2 diabetes, and why does insulin resistance matter?",
                "pubmedqa", "pubmedqa", gold="yes")]
tr = pipe.run(items)
for t in tr:
    print(t.route, t.state.to_dict(), t.final_pids, t.answer, t.cited, t.invalid_cited)
assert tr[1].route == "complex" and tr[1].state.stop_reason == "all_covered"
assert tr[0].invalid_cited == [99] and 99 not in tr[0].cited  # bogus citation flagged
assert '[Doc-99]' not in tr[0].explanation
assert len(set(tr[0].final_pids)) == len(tr[0].final_pids)
assert not ({0, 1} <= set(tr[0].final_pids))                  # near-duplicate removed
print(rrf_fuse([[1, 2, 3], [3, 1, 4]])[:2], validate_citations("x [Doc-1, Doc-7] y", 3), extract_json('```json\n{"a":1}\n```'))
print(evaluate(tr)["diagnostics"])
print("SMOKE OK")

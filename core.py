from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Passage:
    pid: int
    doc_id: str
    title: str
    text: str


@dataclass
class QAItem:
    qid: str
    question: str
    dataset: str                  # pubmedqa | bioasq | mirage
    qtype: str                    # pubmedqa | mcq | yesno | factoid | list | summary
    options: Optional[dict] = None
    gold: Any = None
    meta: dict = field(default_factory=dict)


@dataclass
class EvidenceState:
    """Lightweight evidence-state tracker (covered / unresolved / stop_reason)."""
    covered: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    stop_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"covered": list(self.covered), "unresolved": list(self.unresolved),
                "stop_reason": self.stop_reason}


@dataclass
class Trace:
    item: QAItem
    route: str = "simple"
    subquestions: list = field(default_factory=list)
    state: EvidenceState = field(default_factory=EvidenceState)
    sub_evidence: dict = field(default_factory=dict)     # subquestion -> [pid]
    missing: dict = field(default_factory=dict)          # subquestion -> what is still missing
    iterations: int = 0
    final_pids: list = field(default_factory=list)
    raw_answer: str = ""
    answer: str = ""
    explanation: str = ""
    cited: list = field(default_factory=list)            # valid [Doc-N] numbers
    invalid_cited: list = field(default_factory=list)
    citation_ok: bool = False
    gen_retried: bool = False

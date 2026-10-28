"""Find reporting requirements in regulation text and flag the ones that look
duplicative or outdated.

These are transparent baselines on purpose: every flag carries the signals
that produced it, and the eval harness measures how often they're right.
An optional LLM judge (see judge.py) can confirm candidates.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib import resources
from itertools import combinations
from typing import Any

from regagents.ecfr import Section

OBLIGATION_RE = re.compile(
    r"\b(shall|must|is required to|are required to|will be required to|is responsible for)\b", re.I)
REPORTING_RE = re.compile(
    r"\b(report(s|ed|ing)?|submit(s|ted|tal)?|submission|file[sd]?|filing|notif(y|ies|ication)|"
    r"record(s|keeping)?|maintain(s)?|retain(s)?|furnish(es)?|certif(y|ies|ication)|"
    r"keep (a )?(copy|copies|log))\b", re.I)
# split on sentence ends but not on "U.S.", "e.g.", "i.e.", "No.", "Sec." or decimals
_SENTENCE_RE = re.compile(r"(?<!\bU\.S\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bNo\.)(?<!\bSec\.)(?<=[.;])\s+(?=[A-Z(])")
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
_DEADLINE_RE = re.compile(
    rf"\b(on or before|by|no later than|not later than|until|prior to|before)\s+({_MONTHS})\s+\d{{1,2}},\s+(\d{{4}})", re.I)
_YEAR_REF_RE = re.compile(r"\b(calendar|fiscal|model|crop|reporting) years?\s+(19\d{2}|20[01]\d)\b", re.I)
_WORD_RE = re.compile(r"[a-z][a-z\-]+")
_STOPWORDS = frozenset("""
a an and are as at be been by for from has have in is it its may must of on or such that the
their them then there these this those to under upon was were which who will with within shall
any all each other than not no also if under part section paragraph subpart required requirement
requirements provided accordance pursuant applicable person persons
""".split())


@dataclass
class Requirement:
    id: str
    citation: str  # "40 CFR 60.7(a)(1)"
    section: str
    paragraph: str
    heading: str
    text: str
    url: str
    last_amended: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Flag:
    type: str  # "duplicate" | "outdated"
    requirements: list[Requirement]
    score: float
    reason: str
    signals: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return f"{self.type}:" + "|".join(r.id for r in self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "score": round(self.score, 4), "reason": self.reason,
                "signals": self.signals, "judge": self.judge,
                "requirements": [r.to_dict() for r in self.requirements]}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def is_reporting_requirement(sentence: str) -> bool:
    return bool(OBLIGATION_RE.search(sentence) and REPORTING_RE.search(sentence))


def extract_requirements(sections: list[Section]) -> list[Requirement]:
    out = []
    for s in sections:
        for p_index, para in enumerate(s.paragraphs):
            for s_index, sentence in enumerate(split_sentences(para.text)):
                if is_reporting_requirement(sentence):
                    cite = f"{s.citation}{para.label}"
                    out.append(Requirement(
                        id=f"{s.section}{para.label or f'[{p_index}]'}#{s_index}", citation=cite,
                        section=s.section, paragraph=para.label, heading=s.heading, text=sentence,
                        url=s.url, last_amended=s.last_amended,
                    ))
    return out


# -- duplicates ----------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    words = (w.rstrip("s") if len(w) > 4 else w for w in _WORD_RE.findall(text.lower()))
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


class TfIdf:
    def __init__(self, documents: list[str]):
        self.docs = [Counter(_tokens(d)) for d in documents]
        df = Counter(term for doc in self.docs for term in doc)
        n = max(len(self.docs), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vector(doc) for doc in self.docs]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def cosine(self, i: int, j: int) -> float:
        a, b = self.vectors[i], self.vectors[j]
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def shared_terms(self, i: int, j: int, k: int = 8) -> list[str]:
        a, b = self.vectors[i], self.vectors[j]
        common = sorted(set(a) & set(b), key=lambda t: -(a[t] * b[t]))
        return common[:k]


def similarity(a: str, b: str, corpus: list[str] | None = None) -> float:
    """Cosine similarity of two texts, IDF-weighted over `corpus` (defaults to just the pair)."""
    docs = list(corpus or []) + [a, b]
    model = TfIdf(docs)
    return model.cosine(len(docs) - 2, len(docs) - 1)


def find_duplicates(reqs: list[Requirement], threshold: float = 0.55) -> list[Flag]:
    model = TfIdf([r.text for r in reqs])
    flags = []
    for i, j in combinations(range(len(reqs)), 2):
        a, b = reqs[i], reqs[j]
        if a.section == b.section and a.paragraph == b.paragraph:
            continue  # sentences of one paragraph restate each other by design
        score = model.cosine(i, j)
        if score >= threshold:
            shared = model.shared_terms(i, j)
            flags.append(Flag("duplicate", [a, b], score,
                              f"{a.citation} and {b.citation} impose near-identical reporting obligations",
                              [{"kind": "text_similarity", "cosine": round(score, 4), "shared_terms": shared}]))
    return sorted(flags, key=lambda f: -f.score)


def candidate_pairs(reqs: list[Requirement], floor: float = 0.30) -> list[tuple[Requirement, Requirement, float]]:
    """Pairs above a low similarity floor: the pool hand labels are drawn from."""
    model = TfIdf([r.text for r in reqs])
    pairs = []
    for i, j in combinations(range(len(reqs)), 2):
        a, b = reqs[i], reqs[j]
        if a.section == b.section and a.paragraph == b.paragraph:
            continue
        score = model.cosine(i, j)
        if score >= floor:
            pairs.append((a, b, score))
    return sorted(pairs, key=lambda p: -p[2])


# -- outdated -------------------------------------------------------------------

def load_signals() -> list[dict[str, Any]]:
    raw = resources.files("regagents").joinpath("data/outdated_signals.json").read_text()
    return json.loads(raw)["signals"]


_SIGNALS = None


def outdated_signals(text: str, last_amended: str | None = None, today: date | None = None,
                     stale_years: int = 25) -> list[dict[str, Any]]:
    global _SIGNALS
    if _SIGNALS is None:
        _SIGNALS = [{**s, "re": re.compile(s["pattern"], re.I)} for s in load_signals()]
    today = today or date.today()
    found = []
    for s in _SIGNALS:
        m = s["re"].search(text)
        if m:
            found.append({"kind": s["category"], "match": m.group(0), "note": s["note"], "strong": True})
    for m in _DEADLINE_RE.finditer(text):
        if int(m.group(3)) < today.year - 1:
            found.append({"kind": "past_deadline", "match": m.group(0), "strong": True,
                          "note": "one-time deadline that has passed; requirement may be spent"})
    for m in _YEAR_REF_RE.finditer(text):
        if int(m.group(2)) < today.year - 10:
            found.append({"kind": "dated_year_reference", "match": m.group(0), "strong": True,
                          "note": "obligation tied to a specific past year"})
    if last_amended and int(last_amended[:4]) < today.year - stale_years:
        found.append({"kind": "not_amended", "match": last_amended, "strong": False,
                      "note": f"section not amended in over {stale_years} years"})
    return found


def is_outdated(signals: list[dict[str, Any]]) -> bool:
    return any(s["strong"] for s in signals)


def find_outdated(reqs: list[Requirement], today: date | None = None) -> list[Flag]:
    flags = []
    for r in reqs:
        signals = outdated_signals(r.text, r.last_amended, today)
        if is_outdated(signals):
            strong = [s for s in signals if s["strong"]]
            flags.append(Flag("outdated", [r], float(len(strong)),
                              "; ".join(f"{s['kind']}: '{s['match']}'" for s in strong), signals))
    return sorted(flags, key=lambda f: -f.score)

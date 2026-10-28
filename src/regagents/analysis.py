"""Find reporting requirements in regulation text and flag the ones that look
duplicative or outdated.

Near-identical requirements are split by who they bind. Two requirements in
the same subpart (or one in the part's General Provisions, which apply to
every subpart) can fall on the same regulated party: those are flagged as
`duplicate`. The same sentence repeated across subparts for different source
categories binds different parties, so it isn't redundant reporting; it's
flagged as `boilerplate`, a candidate for harmonizing or centralizing the text.

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
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy import sparse

from regagents.ecfr import Paragraph, Section

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
_REF_RE = re.compile(r"§+\s*(\d+\.\d+[a-z]*)")
_GENERAL_RE = re.compile(r"\bgeneral provisions\b", re.I)
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
    subpart: str = ""
    subpart_heading: str = ""

    @property
    def part(self) -> str:
        return self.section.split(".")[0]

    @property
    def general(self) -> bool:
        """In the part's General Provisions, which apply to every subpart."""
        return bool(_GENERAL_RE.search(self.subpart_heading))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def same_scope(a: Requirement, b: Requirement) -> bool:
    """Whether one regulated party can be bound by both requirements."""
    return a.part == b.part and (a.subpart == b.subpart or a.general or b.general)


@dataclass
class Flag:
    type: str  # "duplicate" | "boilerplate" | "outdated"
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


def sentences(sections: list[Section]) -> Iterator[Requirement]:
    """Every sentence of every paragraph, shaped as a Requirement (whether or not it is one)."""
    for s in sections:
        for p_index, para in enumerate(s.paragraphs):
            for s_index, sentence in enumerate(split_sentences(para.text)):
                yield _sentence(s, para, p_index, s_index, sentence)


def _sentence(s: Section, para: Paragraph, p_index: int, s_index: int, text: str) -> Requirement:
    return Requirement(
        id=f"{s.section}{para.label or f'[{p_index}]'}#{s_index}", citation=f"{s.citation}{para.label}",
        section=s.section, paragraph=para.label, heading=s.heading, text=text, url=s.url,
        last_amended=s.last_amended, subpart=s.subpart, subpart_heading=s.subpart_heading,
    )


def extract_requirements(sections: list[Section]) -> list[Requirement]:
    return [r for r in sentences(sections) if is_reporting_requirement(r.text)]


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
        self.refs = [set(_REF_RE.findall(d)) for d in documents]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def cosine(self, i: int, j: int) -> float:
        """Word cosine, scaled by how much the cross-references agree:
        "records as specified in § 60.395a" vs "... § 60.5420" share every word
        but are different obligations; citing the same section is no evidence
        of duplication on its own, so matching references don't add anything."""
        a, b = self.vectors[i], self.vectors[j]
        if len(a) > len(b):
            a, b = b, a
        words = sum(v * b.get(t, 0.0) for t, v in a.items())
        ra, rb = self.refs[i], self.refs[j]
        return words * (len(ra & rb) / len(ra | rb) if ra or rb else 1.0)

    def pairs(self, floor: float, block: int = 512) -> Iterator[tuple[int, int, float]]:
        """Every pair i < j with cosine(i, j) >= floor.

        Word cosines come from sparse matrix products, a block of rows at a
        time; the cross-reference factor can only lower a score, so it's
        applied to the pairs whose word cosine already clears the floor."""
        vocab = {t: k for k, t in enumerate(self.idf)}
        indptr, indices, data = [0], [], []
        for vec in self.vectors:
            indices += [vocab[t] for t in vec]
            data += vec.values()
            indptr.append(len(indices))
        x = sparse.csr_matrix((np.array(data, dtype=np.float64), indices, indptr),
                              shape=(len(self.vectors), len(vocab)))
        xt = x.T.tocsc()
        for start in range(0, x.shape[0], block):
            words = (x[start:start + block] @ xt).tocoo()
            keep = (words.data >= floor - 1e-9) & (words.col > words.row + start)
            for r, j, w in zip(words.row[keep].tolist(), words.col[keep].tolist(), words.data[keep].tolist()):
                i = r + start
                ra, rb = self.refs[i], self.refs[j]
                score = w * (len(ra & rb) / len(ra | rb) if ra or rb else 1.0)
                if score >= floor:
                    yield i, j, score

    def shared_terms(self, i: int, j: int, k: int = 8) -> list[str]:
        a, b = self.vectors[i], self.vectors[j]
        common = sorted(set(a) & set(b), key=lambda t: -(a[t] * b[t]))
        return common[:k]


def similarity(a: str, b: str, corpus: list[str] | None = None) -> float:
    """Cosine similarity of two texts, IDF-weighted over `corpus` (defaults to just the pair)."""
    docs = list(corpus or []) + [a, b]
    model = TfIdf(docs)
    return model.cosine(len(docs) - 2, len(docs) - 1)


@dataclass
class Similar:
    """Requirement pairs above a similarity floor, each marked same-scope or not.
    Computed once and shared by grouping, boilerplate and the eval sampler."""
    reqs: list[Requirement]
    model: TfIdf
    pairs: list[tuple[int, int, float, bool]]  # (i, j, cosine, same_scope)
    floor: float


def similar_pairs(reqs: list[Requirement], floor: float = 0.30) -> Similar:
    model = TfIdf([r.text for r in reqs])
    pairs = []
    for i, j, score in model.pairs(floor):
        a, b = reqs[i], reqs[j]
        if a.section == b.section and a.paragraph == b.paragraph:
            continue  # sentences of one paragraph restate each other by design
        pairs.append((i, j, score, same_scope(a, b)))
    return Similar(reqs, model, pairs, floor)


def seed_groups(n: int, edges: list[tuple[int, int, float]], threshold: float) -> list[tuple[int, list[int], dict[int, float]]]:
    """Group around seeds: a node joins only if it is within `threshold` of the
    seed itself, which keeps single-linkage chaining from merging unrelated
    text. Returns (seed, members by similarity, the seed's similarities)."""
    neighbours: dict[int, dict[int, float]] = {}
    for i, j, score in edges:
        if score >= threshold:
            neighbours.setdefault(i, {})[j] = score
            neighbours.setdefault(j, {})[i] = score
    assigned: set[int] = set()
    groups = []
    for seed in sorted(neighbours, key=lambda i: (-len(neighbours[i]), i)):
        if seed in assigned:
            continue
        members = sorted((j for j in neighbours[seed] if j not in assigned), key=lambda j: (-neighbours[seed][j], j))
        if not members:
            continue
        assigned.update([seed, *members])
        groups.append((seed, members, neighbours[seed]))
    return groups


def claimed_pairs(similar: Similar, threshold: float) -> set[tuple[int, int]]:
    """The (seed, member) pairs that `find_duplicates` reports at `threshold`, as (min, max)."""
    edges = [(i, j, s) for i, j, s, same in similar.pairs if same]
    return {(min(seed, m), max(seed, m)) for seed, members, _ in seed_groups(len(similar.reqs), edges, threshold)
            for m in members}


def _group_flags(similar: Similar, kind: str, same: bool, threshold: float) -> list[Flag]:
    if threshold < similar.floor:
        raise ValueError(f"threshold {threshold} is below the floor {similar.floor} the pairs were computed at")
    reqs, model = similar.reqs, similar.model
    edges = [(i, j, s) for i, j, s, scope in similar.pairs if scope == same]
    flags = []
    for seed, members, sims_of in seed_groups(len(reqs), edges, threshold):
        group = [reqs[seed], *(reqs[j] for j in members)]
        sims = [sims_of[j] for j in members]
        sections = sorted({r.section for r in group})
        subparts = sorted({f"{r.part} {r.subpart}".strip() for r in group})
        if kind == "duplicate" and group[0].general and not any(r.general for r in group[1:]):
            n, k = len(group) - 1, len(subparts) - 1
            reason = (f"{n} requirement{'s' * (n > 1)} in {k} subpart{'s' * (k > 1)} "
                      f"restate{'s' * (n == 1)} the General Provisions at {group[0].citation}")
        elif kind == "duplicate":
            reason = (f"{group[0].citation} and {group[1].citation} impose near-identical reporting obligations"
                      if len(group) == 2 else
                      f"{len(group) - 1} reporting requirements near-identical to {group[0].citation}, "
                      f"across {len(sections)} sections")
        else:
            reason = (f"near-identical reporting text repeated in {len(subparts)} subparts "
                      f"({len(group)} requirements)")
        flags.append(Flag(kind, group, sum(sims) / len(sims), reason, [{
            "kind": "text_similarity", "seed": group[0].citation, "min_cosine": round(min(sims), 4),
            "shared_terms": model.shared_terms(seed, members[0]),
            "subparts": subparts,
            "members": [{"citation": reqs[j].citation, "cosine": round(sims_of[j], 4)} for j in members],
        }]))
    return sorted(flags, key=lambda f: (-len(f.requirements), -f.score))


def find_duplicates(reqs: list[Requirement], threshold: float = 0.55, similar: Similar | None = None) -> list[Flag]:
    """Groups of near-identical requirements that can bind the same regulated party."""
    return _group_flags(similar or similar_pairs(reqs, threshold), "duplicate", True, threshold)


def find_boilerplate(reqs: list[Requirement], threshold: float = 0.55, similar: Similar | None = None) -> list[Flag]:
    """Groups of near-identical requirements repeated across subparts (different parties)."""
    return _group_flags(similar or similar_pairs(reqs, threshold), "boilerplate", False, threshold)


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

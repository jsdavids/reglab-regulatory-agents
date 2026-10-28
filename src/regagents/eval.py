"""Hand-label evaluation for the flagger.

1. `regagents eval sample` draws a stratified sample of items to label:
   - requirement: is this sentence a reporting or recordkeeping requirement?
     Sentences the extractor kept, plus three strata it dropped: an
     obligation word only, a reporting word only, neither.
   - outdated: extracted requirements the system flagged + ones it didn't
   - duplicate: same-scope requirement pairs above a low similarity floor,
     split into the (seed, member) pairs the report claims and the rest
   Each item records its stratum's sampling weight (stratum size / sampled).
2. You fill in `"label": true/false` for every line (and `notes` if useful).
   See eval/LABELING.md for what counts.
3. `regagents eval score` re-predicts every item and reports weighted
   precision, recall and F1 with stratified-bootstrap 95% intervals.

Weights make recall honest: unflagged items are sampled sparsely, so each
labelled one stands in for many. The outdated and duplicate numbers are
conditional on extraction (they only see sentences the extractor kept), and
duplicate recall is relative to the candidate pool (pairs below the
similarity floor are treated as negatives); say both when you report them.

A duplicate group claims that each member duplicates its seed. Two members
of one group are neither claimed nor denied, so those pairs are left out of
the pool rather than counted as misses.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from regagents.analysis import (OBLIGATION_RE, REPORTING_RE, Requirement, claimed_pairs, is_outdated,
                                is_reporting_requirement, outdated_signals, seed_groups, sentences,
                                similar_pairs)
from regagents.ecfr import EcfrApi, Section

SWEEP = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def _stratified(population: dict[str, list[Any]], n_per_stratum: int, seed: str):
    """A seeded shuffle per stratum, so a larger sample extends a smaller one
    (labels carry over with `eval sample --carry`)."""
    for stratum, items in population.items():
        order = list(range(len(items)))
        random.Random(f"{seed}:{stratum}").shuffle(order)
        chosen = [items[k] for k in sorted(order[:n_per_stratum])]
        weight = len(items) / len(chosen) if chosen else 0.0
        for item in chosen:
            yield stratum, weight, item


def _extraction_stratum(text: str) -> str:
    obligation, reporting = bool(OBLIGATION_RE.search(text)), bool(REPORTING_RE.search(text))
    return {(True, True): "extracted", (True, False): "obligation_only",
            (False, True): "reporting_only", (False, False): "neither"}[(obligation, reporting)]


def build_sample(sections: list[Section], *, n_per_stratum: int = 12, duplicate_threshold: float = 0.55,
                 pair_floor: float = 0.30, seed: int = 0, today: date | None = None) -> list[dict[str, Any]]:
    today = today or date.today()
    items = []

    every = list(sentences(sections))
    by_stratum: dict[str, list[Requirement]] = {k: [] for k in ("extracted", "obligation_only", "reporting_only", "neither")}
    for r in every:
        by_stratum[_extraction_stratum(r.text)].append(r)
    for stratum, weight, r in _stratified(by_stratum, n_per_stratum, f"{seed}:requirement"):
        items.append({"id": f"requirement:{r.id}", "type": "requirement", "stratum": stratum, "weight": weight,
                      "sentence": r.to_dict(), "label": None, "notes": ""})

    reqs = by_stratum["extracted"]
    flagged, unflagged = [], []
    for r in reqs:
        (flagged if is_outdated(outdated_signals(r.text, r.last_amended, today)) else unflagged).append(r)
    for stratum, weight, r in _stratified({"flagged": flagged, "unflagged": unflagged}, n_per_stratum, f"{seed}:outdated"):
        items.append({"id": f"outdated:{r.id}", "type": "outdated", "stratum": stratum, "weight": weight,
                      "as_of": today.isoformat(), "requirement": r.to_dict(), "label": None, "notes": ""})

    similar = similar_pairs(reqs, floor=pair_floor)
    same = [(i, j, score) for i, j, score, scope in similar.pairs if scope]
    claimed = {t: claimed_pairs(similar, t) for t in sorted({*SWEEP, duplicate_threshold})}
    co_members = {(min(a, b), max(a, b)) for seed_, members, _ in seed_groups(len(reqs), same, duplicate_threshold)
                  for k, a in enumerate(members) for b in members[k + 1:]} - claimed[duplicate_threshold]
    pool: dict[str, list[tuple[int, int, float]]] = {"flagged": [], "unflagged": []}
    for i, j, score in same:
        if (i, j) not in co_members:
            pool["flagged" if (i, j) in claimed[duplicate_threshold] else "unflagged"].append((i, j, score))
    for stratum, weight, (i, j, score) in _stratified(pool, n_per_stratum, f"{seed}:duplicate"):
        a, b = reqs[i], reqs[j]
        items.append({"id": f"duplicate:{a.id}|{b.id}", "type": "duplicate", "stratum": stratum,
                      "weight": weight, "score": round(score, 4), "a": a.to_dict(), "b": b.to_dict(),
                      "predicted_at": {f"{t:.2f}": (i, j) in pairs for t, pairs in claimed.items()},
                      "label": None, "notes": ""})
    return items


# -- scoring ------------------------------------------------------------------------

@dataclass
class Metrics:
    precision: float | None
    recall: float | None
    f1: float | None
    n: int
    tp: float
    fp: float
    fn: float


def weighted_metrics(rows: list[tuple[bool, bool, float]]) -> Metrics:
    """rows: (predicted, label, weight)."""
    tp = sum(w for p, y, w in rows if p and y)
    fp = sum(w for p, y, w in rows if p and not y)
    fn = sum(w for p, y, w in rows if not p and y)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return Metrics(precision, recall, f1, len(rows), round(tp, 3), round(fp, 3), round(fn, 3))


def bootstrap_ci(rows: list[tuple[bool, bool, float, str]], stat: Callable[[Metrics], float | None],
                 iterations: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    rng = random.Random(seed)
    by_stratum: dict[str, list] = defaultdict(list)
    for p, y, w, s in rows:
        by_stratum[s].append((p, y, w))
    values = []
    for _ in range(iterations):
        resample = [rng.choice(group) for group in by_stratum.values() for _ in group]
        v = stat(weighted_metrics(resample))
        if v is not None:
            values.append(v)
    if len(values) < iterations * 0.5:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def predict(item: dict[str, Any], duplicate_threshold: float) -> bool:
    if item["type"] == "requirement":
        return is_reporting_requirement(item["sentence"]["text"])
    if item["type"] == "duplicate":
        if "predicted_at" not in item:  # samples drawn before grouping was recorded
            return item["score"] >= duplicate_threshold
        key = f"{duplicate_threshold:.2f}"
        if key not in item["predicted_at"]:
            raise ValueError(f"no predictions recorded at threshold {key}; choose one of {', '.join(item['predicted_at'])}")
        return item["predicted_at"][key]
    r = item["requirement"]
    return is_outdated(outdated_signals(r["text"], r.get("last_amended"), date.fromisoformat(item["as_of"])))


async def judged_predictions(items: list[dict[str, Any]], duplicate_threshold: float) -> dict[str, bool]:
    """Heuristic first, then the LLM judge confirms each heuristic positive."""
    from agentmesh.llm import ClaudeJSON
    from regagents.judge import judge_duplicate, judge_outdated

    llm = ClaudeJSON.from_env()
    if llm is None:
        raise RuntimeError("set AGENTMESH_LLM=1 (and Anthropic credentials) to score with the judge")
    sem = asyncio.Semaphore(8)

    async def one(item):
        if item["type"] == "requirement" or not predict(item, duplicate_threshold):
            return item["id"], predict(item, duplicate_threshold)
        async with sem:
            if item["type"] == "duplicate":
                verdict = await judge_duplicate(llm, item["a"], item["b"])
            else:
                r = item["requirement"]
                verdict = await judge_outdated(llm, r, outdated_signals(r["text"], r.get("last_amended"),
                                                                        date.fromisoformat(item["as_of"])))
        return item["id"], bool(verdict["verdict"])

    return dict(await asyncio.gather(*(one(i) for i in items)))


def score(items: list[dict[str, Any]], duplicate_threshold: float = 0.55,
          predictions: dict[str, bool] | None = None) -> dict[str, Any]:
    labelled = [i for i in items if isinstance(i.get("label"), bool)]
    report: dict[str, Any] = {"labelled": len(labelled), "unlabelled": len(items) - len(labelled)}
    for kind in ("requirement", "outdated", "duplicate"):
        subset = [i for i in labelled if i["type"] == kind]
        rows = [((predictions[i["id"]] if predictions else predict(i, duplicate_threshold)),
                 i["label"], float(i["weight"]), i["stratum"]) for i in subset]
        m = weighted_metrics([r[:3] for r in rows])
        report[kind] = {
            **m.__dict__,
            "precision_ci95": bootstrap_ci(rows, lambda x: x.precision),
            "recall_ci95": bootstrap_ci(rows, lambda x: x.recall),
        }
    dupes = [i for i in labelled if i["type"] == "duplicate"]
    if dupes and predictions is None:
        sweep = []
        for t in SWEEP:
            m = weighted_metrics([(predict(i, t), i["label"], float(i["weight"])) for i in dupes])
            sweep.append({"threshold": t, "precision": m.precision, "recall": m.recall, "f1": m.f1})
        report["duplicate_threshold_sweep"] = sweep
    return report


# -- CLI glue -------------------------------------------------------------------------

def sample_command(title: int, parts: list[str], out: Path, n_per_stratum: int, seed: int,
                   date_: str | None = None, carry: Path | None = None) -> int:
    api = EcfrApi()
    sections = [s for p in parts for s in api.part(title, p, date_)]
    items = build_sample(sections, n_per_stratum=n_per_stratum, seed=seed)
    if carry:
        done = {i["id"]: i for i in load_items(carry) if isinstance(i.get("label"), bool)}
        for item in items:
            if item["id"] in done:
                item.update(label=done[item["id"]]["label"], notes=done[item["id"]].get("notes", ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    return len(items)


def load_items(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_items(path: Path, items: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(i) + "\n" for i in items))
    tmp.replace(path)


QUESTIONS = {
    "requirement": "Does this sentence oblige a regulated party to report, notify, submit, record or keep records?",
    "outdated": "Is this requirement outdated as of {as_of} (obsolete technology, defunct agency, spent deadline)?",
    "duplicate": "Would one party bound by both have to report or record substantially the same thing twice?",
}


def _show(item: dict[str, Any]) -> str:
    def req(r: dict[str, Any]) -> str:
        where = f" [{r['subpart_heading']}]" if r.get("subpart_heading") else ""
        return f"  {r['citation']}{where}\n  {r['url']}\n  \"{r['text']}\""
    body = {"requirement": lambda: req(item["sentence"]), "outdated": lambda: req(item["requirement"]),
            "duplicate": lambda: f"{req(item['a'])}\n\n{req(item['b'])}\n  (cosine {item['score']})"}[item["type"]]()
    return f"{item['type'].upper()}: {QUESTIONS[item['type']].format(**item)}\n\n{body}"


def label_command(path: Path, read: Callable[[str], str] = input, write: Callable[[str], None] = print) -> int:
    """Prompt for each unlabelled item; saves after every answer, so it can be stopped and resumed."""
    items = load_items(path)
    todo = [i for i in items if not isinstance(i.get("label"), bool)]
    for n, item in enumerate(todo, 1):
        write(f"\n--- {n}/{len(todo)}  {item['id']}\n{_show(item)}")
        while True:
            answer = read("y/n [note], s to skip, q to quit: ").strip()
            key, _, note = answer.partition(" ")
            if key.lower() in ("y", "n", "s", "q"):
                break
        if key.lower() == "q":
            break
        if key.lower() == "s":
            continue
        item["label"] = key.lower() == "y"
        item["notes"] = note.strip() or item.get("notes", "")
        save_items(path, items)
    return sum(isinstance(i.get("label"), bool) for i in items)


def agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Cohen's kappa between two labellers on the items both labelled."""
    theirs = {i["id"]: i["label"] for i in b if isinstance(i.get("label"), bool)}
    pairs = [(i["label"], theirs[i["id"]]) for i in a if isinstance(i.get("label"), bool) and i["id"] in theirs]
    if not pairs:
        return {"n": 0, "observed": None, "kappa": None}
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"n": n, "observed": round(observed, 4), "kappa": None if kappa is None else round(kappa, 4),
            "disagreements": [i["id"] for i in a if i["id"] in theirs and isinstance(i.get("label"), bool)
                              and i["label"] != theirs[i["id"]]]}

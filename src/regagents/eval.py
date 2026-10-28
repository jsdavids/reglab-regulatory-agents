"""Hand-label evaluation for the flagger.

1. `regagents eval sample` draws a stratified sample of items to label:
   - outdated: requirements the system flagged + requirements it didn't
   - duplicate: requirement pairs above a low similarity floor, split into
     pairs the system flags and pairs it doesn't
   Each item records its stratum's sampling weight (stratum size / sampled).
2. You fill in `"label": true/false` for every line (and `notes` if useful).
3. `regagents eval score` re-predicts every item and reports weighted
   precision, recall and F1 with stratified-bootstrap 95% intervals.

Weights make recall honest: unflagged requirements are sampled sparsely, so
each labelled one stands in for many. Duplicate recall is relative to the
candidate pool (pairs below the similarity floor are treated as negatives);
say so when you report it.
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

from regagents.analysis import (Requirement, candidate_pairs, extract_requirements, is_outdated,
                                outdated_signals)
from regagents.ecfr import EcfrApi


def _stratified(population: dict[str, list[Any]], n_per_stratum: int, rng: random.Random):
    for stratum, items in population.items():
        chosen = items if len(items) <= n_per_stratum else rng.sample(items, n_per_stratum)
        weight = len(items) / len(chosen) if chosen else 0.0
        for item in chosen:
            yield stratum, weight, item


def build_sample(reqs: list[Requirement], *, n_per_stratum: int = 25, duplicate_threshold: float = 0.55,
                 pair_floor: float = 0.30, seed: int = 0, today: date | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    today = today or date.today()
    items = []

    flagged, unflagged = [], []
    for r in reqs:
        (flagged if is_outdated(outdated_signals(r.text, r.last_amended, today)) else unflagged).append(r)
    for stratum, weight, r in _stratified({"flagged": flagged, "unflagged": unflagged}, n_per_stratum, rng):
        items.append({"id": f"outdated:{r.id}", "type": "outdated", "stratum": stratum, "weight": weight,
                      "as_of": today.isoformat(), "requirement": r.to_dict(), "label": None, "notes": ""})

    pairs = candidate_pairs(reqs, floor=pair_floor)
    above = [p for p in pairs if p[2] >= duplicate_threshold]
    below = [p for p in pairs if p[2] < duplicate_threshold]
    for stratum, weight, (a, b, score) in _stratified({"flagged": above, "unflagged": below}, n_per_stratum, rng):
        items.append({"id": f"duplicate:{a.id}|{b.id}", "type": "duplicate", "stratum": stratum,
                      "weight": weight, "score": round(score, 4), "a": a.to_dict(), "b": b.to_dict(),
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
    if item["type"] == "duplicate":
        return item["score"] >= duplicate_threshold
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
        if not predict(item, duplicate_threshold):
            return item["id"], False
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
    for kind in ("outdated", "duplicate"):
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
        for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            m = weighted_metrics([(i["score"] >= t, i["label"], float(i["weight"])) for i in dupes])
            sweep.append({"threshold": t, "precision": m.precision, "recall": m.recall, "f1": m.f1})
        report["duplicate_threshold_sweep"] = sweep
    return report


# -- CLI glue -------------------------------------------------------------------------

def sample_command(title: int, parts: list[str], out: Path, n_per_stratum: int, seed: int,
                   date_: str | None = None) -> int:
    api = EcfrApi()
    sections = [s for p in parts for s in api.part(title, p, date_)]
    items = build_sample(extract_requirements(sections), n_per_stratum=n_per_stratum, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    return len(items)


def load_items(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

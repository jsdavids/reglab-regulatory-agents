import random
from datetime import date

from ecfr_mock import transport

from regagents.analysis import extract_requirements
from regagents.ecfr import EcfrApi
from regagents.eval import bootstrap_ci, build_sample, score, weighted_metrics


def test_weighted_metrics_hand_computed():
    # weights stand in for unsampled population: one unflagged FN of weight 3
    rows = [(True, True, 1.0), (True, False, 1.0), (False, True, 3.0), (False, False, 3.0)]
    m = weighted_metrics(rows)
    assert m.precision == 0.5
    assert m.recall == 0.25
    assert round(m.f1, 4) == round(2 * 0.5 * 0.25 / 0.75, 4)


def test_metrics_undefined_without_positives():
    m = weighted_metrics([(False, False, 1.0)])
    assert m.precision is None and m.recall is None and m.f1 is None


def test_bootstrap_interval_contains_point_estimate():
    rng = random.Random(1)
    rows = [(p, (p if rng.random() < 0.8 else not p), 1.0, "s") for p in [rng.random() < 0.5 for _ in range(200)]]
    point = weighted_metrics([r[:3] for r in rows]).precision
    lo, hi = bootstrap_ci(rows, lambda m: m.precision, iterations=500)
    assert lo <= point <= hi and hi - lo < 0.3


def test_sample_then_score_end_to_end():
    reqs = extract_requirements(EcfrApi(cache_dir="", transport=transport()).part(99, "900"))
    items = build_sample(reqs, n_per_stratum=50, today=date(2026, 10, 28))
    assert {i["type"] for i in items} == {"outdated", "duplicate"}
    assert all(i["label"] is None for i in items)
    # an oracle labeller that agrees with the system gets perfect precision/recall
    for i in items:
        i["label"] = i["stratum"] == "flagged"
    report = score(items)
    assert report["outdated"]["precision"] == 1.0 and report["outdated"]["recall"] == 1.0
    assert report["duplicate"]["precision"] == 1.0
    assert report["unlabelled"] == 0

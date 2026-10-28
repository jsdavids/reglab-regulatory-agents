import random
from datetime import date

from ecfr_mock import transport

from regagents.analysis import Requirement, similar_pairs
from regagents.ecfr import EcfrApi, Paragraph, Section
import json

from regagents.eval import (agreement, bootstrap_ci, build_sample, label_command, load_items, predict, score,
                            weighted_metrics)


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
    sections = EcfrApi(cache_dir="", transport=transport()).part(99, "900")
    items = build_sample(sections, n_per_stratum=50, today=date(2026, 10, 28))
    assert {i["type"] for i in items} == {"requirement", "outdated", "duplicate"}
    assert {i["stratum"] for i in items if i["type"] == "requirement"} == {
        "extracted", "obligation_only", "reporting_only", "neither"}
    assert all(i["label"] is None for i in items)
    # the cross-subpart boilerplate pair isn't in the duplicate pool
    assert [i["id"] for i in items if i["type"] == "duplicate"] == ["duplicate:900.2(a)#0|900.3(a)#0"]
    # an oracle labeller that agrees with the system gets perfect precision/recall
    for i in items:
        i["label"] = i["stratum"] in ("flagged", "extracted")
    report = score(items)
    for kind in ("requirement", "outdated"):
        assert report[kind]["precision"] == 1.0 and report[kind]["recall"] == 1.0
    assert report["duplicate"]["precision"] == 1.0
    assert report["unlabelled"] == 0


def test_requirement_recall_counts_dropped_sentences():
    sections = EcfrApi(cache_dir="", transport=transport()).part(99, "900")
    items = build_sample(sections, n_per_stratum=50, today=date(2026, 10, 28))
    reqs = [i for i in items if i["type"] == "requirement"]
    for i in reqs:  # "Records ... may be kept electronically" is a recordkeeping rule the regex drops
        i["label"] = i["stratum"] == "extracted" or "may be kept electronically" in i["sentence"]["text"]
    m = score(reqs)["requirement"]
    assert m["precision"] == 1.0 and m["recall"] < 1.0 and m["fn"] == 1.0


def test_group_members_are_not_counted_as_misses():
    # seed 1.1 links to 1.2 and 1.3; 1.2 vs 1.3 is a co-member pair the report makes no claim about
    texts = {"1.1": "The owner or operator shall submit an annual emissions report to the Administrator.",
             "1.2": "The owner or operator shall submit an annual emissions report to the Administrator by March 31.",
             "1.3": "The owner or operator shall submit an annual emissions report to the Administrator electronically."}
    sections = [Section(1, "1", n, "x", [Paragraph("(a)", t)]) for n, t in texts.items()]
    items = [i for i in build_sample(sections, n_per_stratum=50, today=date(2026, 10, 28)) if i["type"] == "duplicate"]
    reqs = [Requirement(**i[k]) for i in items for k in ("a", "b")]
    assert len(similar_pairs(list({r.id: r for r in reqs}.values())).pairs) == 3
    assert len(items) == 2 and all(i["stratum"] == "flagged" and predict(i, 0.55) for i in items)
    assert set(items[0]["predicted_at"]) >= {"0.35", "0.55", "0.80"}


def test_label_command_saves_each_answer_and_resumes(tmp_path):
    sections = EcfrApi(cache_dir="", transport=transport()).part(99, "900")
    path = tmp_path / "to_label.jsonl"
    path.write_text("".join(json.dumps(i) + "\n" for i in build_sample(sections, n_per_stratum=1, today=date(2026, 10, 28))))
    answers = iter(["maybe", "y cites 900.2", "s", "n", "q"])
    shown = []
    assert label_command(path, read=lambda _: next(answers), write=shown.append) == 2
    first, second, third = load_items(path)[:3]
    assert (first["label"], first["notes"]) == (True, "cites 900.2") and second["label"] is None and third["label"] is False
    assert "REQUIREMENT:" in shown[0] and "Subpart" in shown[0]
    assert label_command(path, read=lambda _: "q", write=shown.append) == 2  # resumes at the first unlabelled item
    assert shown[-1].split()[1] == "1/" + str(len(load_items(path)) - 2)


def test_cohens_kappa():
    a = [{"id": str(k), "label": v} for k, v in enumerate([True, True, False, False])]
    b = [{"id": str(k), "label": v} for k, v in enumerate([True, False, False, False])] + [{"id": "x", "label": True}]
    out = agreement(a, b)
    assert out["n"] == 4 and out["observed"] == 0.75 and out["kappa"] == 0.5 and out["disagreements"] == ["1"]


def test_larger_sample_extends_smaller_one():
    sections = [Section(1, "1", f"1.{n}", "x", [Paragraph("(a)", f"Sentence number {n} is about widgets.")])
                for n in range(1, 60)]
    small = {i["id"] for i in build_sample(sections, n_per_stratum=5, today=date(2026, 10, 28))}
    large = {i["id"] for i in build_sample(sections, n_per_stratum=10, today=date(2026, 10, 28))}
    assert len(small) == 5 and len(large) == 10 and small < large

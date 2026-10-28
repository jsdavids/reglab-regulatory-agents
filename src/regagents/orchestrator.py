"""Orchestrator: retriever -> flagger -> citer, all over A2A through the gateway.
It holds no tool grants of its own; it can only delegate."""

from __future__ import annotations

import time
from typing import Any

from agentmesh import MeshClient


async def review(mesh: MeshClient, title: int, parts: list[str], date: str | None = None,
                 duplicate_threshold: float = 0.55) -> dict[str, Any]:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    corpus = await mesh.call_agent("retriever", {"title": title, "parts": parts, "date": date})
    timings["retrieve_s"] = time.perf_counter() - t

    t = time.perf_counter()
    flagged = await mesh.call_agent("flagger", {"sections": corpus["sections"],
                                                "duplicate_threshold": duplicate_threshold})
    timings["flag_s"] = time.perf_counter() - t

    t = time.perf_counter()
    cited = await mesh.call_agent("citer", {"title": title, "flags": flagged["flags"]})
    timings["cite_s"] = time.perf_counter() - t

    as_of = next((s.get("date") for s in corpus["sections"] if s.get("date")), None)
    return {
        "title": title, "parts": parts, "as_of": as_of,
        "sections": len(corpus["sections"]),
        "requirements_found": flagged["requirements_found"],
        "judge": flagged.get("judge"),
        "flags": cited["flags"],
        "unverified": cited["unverified"],
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Reporting-requirement review: {report['title']} CFR parts {', '.join(report['parts'])}",
        "",
        f"eCFR text as of {report['as_of']}. {report['sections']} sections, "
        f"{report['requirements_found']} reporting requirements, {len(report['flags'])} verified flags"
        + (f", {len(report['unverified'])} dropped because their quotes could not be verified." if report["unverified"] else "."),
        "",
    ]
    for kind, title in [("duplicate", "Possible duplicates"), ("outdated", "Possibly outdated")]:
        items = [f for f in report["flags"] if f["type"] == kind]
        lines += [f"## {title} ({len(items)})", ""]
        for f in items:
            lines.append(f"- **{f['reason']}** (score {f['score']})")
            for c in f["citations"]:
                lines.append(f"  - [{c['citation']}]({c['url']}): \"{c['quote']}\"")
            if f.get("judge"):
                lines.append(f"  - judge: {f['judge']['rationale']}")
        lines.append("")
    return "\n".join(lines)

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
    flagged = await mesh.call_agent("flagger", {"requirements": corpus["requirements"],
                                                "duplicate_threshold": duplicate_threshold})
    timings["flag_s"] = time.perf_counter() - t

    t = time.perf_counter()
    cited = await mesh.call_agent("citer", {"title": title, "flags": flagged["flags"]})
    timings["cite_s"] = time.perf_counter() - t

    return {
        "title": title, "parts": parts, "as_of": corpus["as_of"],
        "sections": corpus["sections"],
        "requirements_found": flagged["requirements_found"],
        "found": flagged.get("found", {}),
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
    sections = [
        ("duplicate", "Possible duplicates",
         "Near-identical requirements that can bind the same regulated party: same subpart, "
         "or restating the part's General Provisions."),
        ("boilerplate", "Repeated across subparts",
         "The same text imposed on different source categories. Not redundant for any one party; "
         "candidates for harmonizing or moving into the General Provisions."),
        ("outdated", "Possibly outdated", ""),
    ]
    found = report.get("found", {})
    for kind, title, blurb in sections:
        items = [f for f in report["flags"] if f["type"] == kind]
        total = f" of {found[kind]} found" if found.get(kind, len(items)) > len(items) else ""
        lines += [f"## {title} ({len(items)}{total})", ""]
        if blurb:
            lines += [blurb, ""]
        for f in items:
            lines.append(f"- **{f['reason']}** (score {f['score']})")
            for c in f["citations"]:
                lines.append(f"  - [{c['citation']}]({c['url']}): \"{c['quote']}\"")
            if f.get("judge"):
                lines.append(f"  - judge: {f['judge']['rationale']}")
        lines.append("")
    return "\n".join(lines)

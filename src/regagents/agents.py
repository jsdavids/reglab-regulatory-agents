"""The three specialist agents. Each is an A2A server; each reaches eCFR only
through the gateway, with the tool grants in mesh.yaml:

    retriever  ecfr: get_part, search, list_titles   pulls regulation text, extracts requirements
    flagger    (no tools)                            groups duplicates, flags outdated requirements
    citer      ecfr: get_section                     re-reads each cited section and verifies quotes
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agentmesh import AgentContext, MeshClient, Skill, build_agent_app
from agentmesh.llm import ClaudeJSON
from regagents.analysis import Requirement, extract_requirements, find_duplicates, find_outdated
from regagents.ecfr import Section
from regagents.judge import judge_duplicate, judge_outdated


# -- retriever --------------------------------------------------------------------

async def retrieve(payload: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    """Pull the parts and extract their reporting requirements here, next to
    the text: a CFR part is 9-18 MB of JSON, its requirements about a quarter of that."""
    title = int(payload["title"])
    parts = payload.get("parts") or [payload["part"]]
    sections = []
    async with ctx.mesh.mcp("ecfr") as ecfr:
        for part in parts:
            sections += await ecfr.call("get_part", title=title, part=str(part), date=payload.get("date"))
    reqs = extract_requirements([Section.from_dict(s) for s in sections])
    return {"title": title, "parts": [str(p) for p in parts], "sections": len(sections),
            "as_of": next((s.get("date") for s in sections if s.get("date")), None),
            "requirements": [r.to_dict() for r in reqs]}


def retriever(url: str = "http://127.0.0.1:9201", mesh: MeshClient | None = None):
    return build_agent_app(
        name="retriever", description="Pulls current CFR text from eCFR", url=url, handler=retrieve,
        skills=[Skill("fetch_parts", "Fetch CFR parts", "Returns sections and paragraphs for CFR parts",
                      tags=["ecfr", "retrieval"], examples=['{"title": 40, "parts": ["60"]}'])],
        mesh=mesh or MeshClient.from_env(),
    )


# -- flagger ------------------------------------------------------------------------

async def flag(payload: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    reqs = [Requirement(**r) for r in payload["requirements"]]
    cap = int(payload.get("max_flags", 200))  # per type, so duplicates can't crowd out outdated flags
    flags = find_duplicates(reqs, threshold=float(payload.get("duplicate_threshold", 0.55)))[:cap]
    flags += find_outdated(reqs)[:cap]

    llm = ClaudeJSON.from_env()
    if llm is not None:
        sem = asyncio.Semaphore(8)

        async def review(f):
            async with sem:
                if f.type == "duplicate":
                    f.judge = await judge_duplicate(llm, f.requirements[0].to_dict(), f.requirements[1].to_dict())
                else:
                    f.judge = await judge_outdated(llm, f.requirements[0].to_dict(), f.signals)

        await asyncio.gather(*(review(f) for f in flags))
        flags = [f for f in flags if f.judge["verdict"]]

    return {"requirements_found": len(reqs), "flags": [f.to_dict() for f in flags],
            "judge": llm.model if llm else None}


def flagger(url: str = "http://127.0.0.1:9202", mesh: MeshClient | None = None):
    return build_agent_app(
        name="flagger", description="Flags duplicative or outdated reporting requirements", url=url,
        handler=flag, mesh=mesh or MeshClient.from_env(),
        skills=[Skill("flag_requirements", "Flag requirements",
                      "Extracts reporting requirements and flags duplicates and outdated ones",
                      tags=["analysis"])],
    )


# -- citer -----------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(section: dict[str, Any], req: Requirement) -> bool:
    quote = _norm(req.text)
    for para in section["paragraphs"]:
        if para["label"] == req.paragraph and quote in _norm(para["text"]):
            return True
    return False


async def cite(payload: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    flags = payload["flags"]
    title = int(payload["title"])
    needed = sorted({r["section"] for f in flags for r in f["requirements"]})
    fetched: dict[str, dict[str, Any] | None] = {}
    async with ctx.mesh.mcp("ecfr") as ecfr:
        for section in needed:
            try:
                fetched[section] = await ecfr.call("get_section", title=title, section=section)
            except Exception:
                fetched[section] = None
    verified, unverified = [], []
    for f in flags:
        citations = []
        for r in f["requirements"]:
            req = Requirement(**r)
            source = fetched.get(req.section)
            citations.append({"citation": req.citation, "url": req.url, "quote": req.text,
                              "as_of": source.get("date") if source else None,
                              "verified": bool(source and verify_quote(source, req))})
        f = {**f, "citations": citations}
        (verified if all(c["verified"] for c in citations) else unverified).append(f)
    return {"flags": verified, "unverified": unverified}


def citer(url: str = "http://127.0.0.1:9203", mesh: MeshClient | None = None):
    return build_agent_app(
        name="citer", description="Verifies every quoted requirement against the live eCFR text",
        url=url, handler=cite, mesh=mesh or MeshClient.from_env(),
        skills=[Skill("cite", "Cite and verify", "Attaches exact citations and drops quotes that don't match",
                      tags=["citations"])],
    )

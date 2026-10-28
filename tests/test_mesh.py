"""Full mesh: eCFR MCP server (mocked upstream) + 3 agents + gateway."""

import pytest
from ecfr_mock import transport

from agentmesh import ToolCallError, mcp_app, verify_chain
from agentmesh.testing import AgentSpec, LocalMesh
from regagents import agents
from regagents.ecfr import EcfrApi
from regagents.mcp_ecfr import build_server
from regagents.orchestrator import review, to_markdown

PRINCIPALS = {
    "orchestrator": {"agents": ["retriever", "flagger", "citer"]},
    "retriever": {"mcp": {"ecfr": ["get_part", "search", "list_titles"]}},
    "citer": {"mcp": {"ecfr": ["get_section"]}},
    "flagger": {},
}


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    api = EcfrApi(cache_dir="", transport=transport())
    with LocalMesh(
        principals=PRINCIPALS,
        mcp_servers={"ecfr": lambda: mcp_app(build_server(api))},
        agents={
            "retriever": AgentSpec(agents.retriever, "retriever"),
            "flagger": AgentSpec(agents.flagger, "flagger"),
            "citer": AgentSpec(agents.citer, "citer"),
        },
        audit_path=tmp_path_factory.mktemp("audit") / "gw.jsonl",
    ) as m:
        yield m


async def test_review_pipeline(mesh):
    report = await review(mesh.client("orchestrator"), 99, ["900"])
    assert report["as_of"] == "2026-10-20"
    types = {f["type"] for f in report["flags"]}
    assert types == {"duplicate", "boilerplate", "outdated"}
    assert report["found"] == {"duplicate": 1, "boilerplate": 1, "outdated": 3}
    assert report["unverified"] == []
    for f in report["flags"]:
        for c in f["citations"]:
            assert c["verified"] and c["url"].startswith("https://www.ecfr.gov/current/title-99/section-900.")
    md = to_markdown(report)
    assert "99 CFR 900.2(a)" in md and "facsimile" in md and "## Repeated across subparts (1)" in md


async def test_citer_drops_fabricated_quotes(mesh):
    fake = {"type": "outdated", "requirements": [{
        "id": "x", "citation": "99 CFR 900.3(a)", "section": "900.3", "paragraph": "(a)", "heading": "",
        "text": "The owner shall submit reports by telegraph.", "url": "u", "last_amended": None}]}
    out = await mesh.client("orchestrator").call_agent("citer", {"title": 99, "flags": [fake]})
    assert out["flags"] == [] and len(out["unverified"]) == 1


async def test_least_privilege_is_enforced(mesh):
    with pytest.raises(ToolCallError):
        await mesh.client("citer").call_tool("ecfr", "get_part", title=99, part="900")
    with pytest.raises(ToolCallError):
        await mesh.client("flagger").call_tool("ecfr", "get_section", title=99, section="900.1")
    with pytest.raises(ToolCallError):
        await mesh.client("orchestrator").call_tool("ecfr", "search", query="report")
    async with mesh.client("citer").mcp("ecfr") as tools:
        assert await tools.names() == ["get_section"]


async def test_audit_trail(mesh):
    await review(mesh.client("orchestrator"), 99, ["900"])
    assert verify_chain(mesh.gateway.audit.path).ok
    calls = [(r["principal"], r.get("tool")) for r in mesh.gateway.audit.records
             if r["decision"] == "allowed" and r.get("tool")]
    assert ("retriever", "get_part") in calls and ("citer", "get_section") in calls

"""MCP server exposing the eCFR API as tools."""

from __future__ import annotations

from functools import partial
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from agentmesh import mcp_app
from regagents.ecfr import EcfrApi


def build_server(api: EcfrApi) -> MCPServer:
    server = MCPServer("ecfr", instructions="Read-only access to the Electronic Code of Federal Regulations.")

    async def run(fn, *args, **kwargs):
        return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))

    @server.tool()
    async def list_titles() -> list[dict[str, Any]]:
        """List CFR titles with the date each is current as of."""
        titles = await run(api.titles)
        return [{"number": t["number"], "name": t["name"], "up_to_date_as_of": t.get("up_to_date_as_of")}
                for t in titles]

    @server.tool()
    async def get_part(title: int, part: str, date: str | None = None) -> list[dict[str, Any]]:
        """Full text of a CFR part as sections with labelled paragraphs.
        `date` is YYYY-MM-DD; omit it for the current version."""
        return [s.to_dict() for s in await run(api.part, title, part, date)]

    @server.tool()
    async def get_section(title: int, section: str, date: str | None = None) -> dict[str, Any]:
        """Full text of one section, e.g. title=40, section="60.7"."""
        return (await run(api.section, title, section, date)).to_dict()

    @server.tool()
    async def search(query: str, title: int | None = None, per_page: int = 20) -> list[dict[str, Any]]:
        """Full-text search across current regulations, optionally within one title."""
        return await run(api.search, query, title, per_page)

    return server


def app():
    return mcp_app(build_server(EcfrApi()))

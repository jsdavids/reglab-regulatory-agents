"""httpx MockTransport that serves the synthetic eCFR fixtures."""

import json
from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures"
DATE = "2026-10-20"


def _section_xml(section: str) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring((FIXTURES / "part_900.xml").read_bytes())
    for div in root.iter("DIV8"):
        if div.get("N", "").replace("§", "").strip() == section:
            return ET.tostring(div, encoding="unicode")
    return "<DIV8/>"


def handler(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, request.url.params
    if path == "/api/versioner/v1/titles.json":
        return httpx.Response(200, json={"titles": [{"number": 99, "name": "Synthetic", "up_to_date_as_of": DATE}]})
    if path == f"/api/versioner/v1/full/{DATE}/title-99.xml" and params.get("part") == "900":
        if "section" in params:
            return httpx.Response(200, text=_section_xml(params["section"]))
        return httpx.Response(200, text=(FIXTURES / "part_900.xml").read_text())
    if path == "/api/versioner/v1/versions/title-99.json":
        return httpx.Response(200, text=(FIXTURES / "versions_99.json").read_text())
    if path == "/api/search/v1/results":
        return httpx.Response(200, json={"results": [{"hierarchy": {"title": "99", "part": "900", "section": "900.3"},
                                                      "headings": {"section": "Annual reports."},
                                                      "full_text_excerpt": "shall submit an <strong>annual report</strong>"}]})
    return httpx.Response(404, json={"error": f"no fixture for {request.url}"})


def transport() -> httpx.MockTransport:
    return httpx.MockTransport(handler)

"""Client for the public eCFR API (https://www.ecfr.gov/developers/documentation/api/v1).

Responses are cached on disk because CFR text for a given date never
changes, and the API is a shared public resource.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://www.ecfr.gov"
USER_AGENT = "regagents/0.1 (research prototype; +https://github.com/)"

# Paragraph designations nest (a) -> (1) -> (i) -> (A) -> (<i>1</i>) -> (<i>i</i>);
# italics are lost in plain text, so levels 4/5 are inferred from context.
_LABEL_RE = re.compile(r"^\s*\(([a-zA-Z]{1,5}|\d{1,3})\)")
_ROMAN_RE = re.compile(r"^(?=.)x{0,3}(ix|iv|v?i{0,3})$")  # i..xxxix

@dataclass
class Paragraph:
    label: str  # e.g. "(b)(2)(i)", "" for undesignated text
    text: str


@dataclass
class Section:
    title: int
    part: str
    section: str  # "60.7"
    heading: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    date: str | None = None
    last_amended: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.title} CFR {self.section}"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/current/title-{self.title}/section-{self.section}"

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(citation=self.citation, url=self.url)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Section:
        return cls(title=int(d["title"]), part=str(d["part"]), section=str(d["section"]),
                   heading=d.get("heading", ""), date=d.get("date"), last_amended=d.get("last_amended"),
                   paragraphs=[Paragraph(**p) for p in d.get("paragraphs", [])])


def _level(token: str, stack: list[tuple[int, str]]) -> int:
    deepest = stack[-1][0] if stack else -1
    if token.isdigit():
        return 4 if deepest >= 3 else 1
    if token.isupper():
        return 3
    letter = next((t for lvl, t in stack if lvl == 0), "")
    follows_letter = len(token) == 1 and len(letter) == 1 and ord(token) == ord(letter) + 1
    if _ROMAN_RE.match(token) and deepest >= 1 and not follows_letter:
        # (i) after (h)(2) is the next letter; after (1) or (ii) it is a roman numeral
        return 5 if deepest >= 4 else 2
    return 0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_part_xml(xml: str, title: int, part: str, date: str | None = None) -> list[Section]:
    root = ET.fromstring(xml)
    sections = []
    for div in root.iter():
        if div.tag.startswith("DIV") and div.get("TYPE") == "SECTION":
            number = (div.get("N") or "").replace("§", "").strip()
            head = div.find("HEAD")
            heading = _clean("".join(head.itertext())) if head is not None else ""
            heading = re.sub(r"^§+\s*[\d.]+\s*", "", heading)
            paragraphs, stack = [], []
            for p in div.iter("P"):
                text = _clean("".join(p.itertext()))
                if not text:
                    continue
                labels, rest = [], text
                while (m := _LABEL_RE.match(rest)):
                    labels.append(m.group(1))
                    rest = rest[m.end():]
                for token in labels:
                    level = _level(token, stack)
                    stack = [entry for entry in stack if entry[0] < level] + [(level, token)]
                label = "".join(f"({t})" for _, t in stack) if labels else ""
                paragraphs.append(Paragraph(label=label, text=text))
            sections.append(Section(title=title, part=part, section=number, heading=heading,
                                    paragraphs=paragraphs, date=date))
    return sections


class EcfrApi:
    def __init__(self, base_url: str = BASE_URL, cache_dir: str | Path | None = None,
                 transport: httpx.BaseTransport | None = None):
        cache = cache_dir if cache_dir is not None else os.environ.get("REGAGENTS_CACHE", "cache/ecfr")
        self.cache_dir = Path(cache) if cache else None
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=60.0,
                                 headers={"user-agent": USER_AGENT}, follow_redirects=True)

    def _get(self, path: str, params: dict[str, Any] | None = None, cacheable: bool = True) -> str:
        key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:32]
        cached = self.cache_dir / key if self.cache_dir else None
        if cacheable and cached and cached.exists():
            return cached.read_text()
        resp = self.http.get(path, params=params)
        resp.raise_for_status()
        if cacheable and cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(resp.text)
        return resp.text

    def titles(self) -> list[dict[str, Any]]:
        # not cached: up_to_date_as_of moves daily
        return json.loads(self._get("/api/versioner/v1/titles.json", cacheable=False))["titles"]

    def current_date(self, title: int) -> str:
        for t in self.titles():
            if int(t["number"]) == int(title):
                return t["up_to_date_as_of"]
        raise ValueError(f"unknown CFR title {title}")

    def part(self, title: int, part: str, date: str | None = None) -> list[Section]:
        date = date or self.current_date(title)
        xml = self._get(f"/api/versioner/v1/full/{date}/title-{title}.xml", {"part": part})
        sections = parse_part_xml(xml, title, str(part), date)
        amended = self.last_amended(title, part)
        for s in sections:
            s.last_amended = amended.get(s.section)
        return sections

    def section(self, title: int, section: str, date: str | None = None) -> Section:
        date = date or self.current_date(title)
        part = section.split(".")[0]
        xml = self._get(f"/api/versioner/v1/full/{date}/title-{title}.xml", {"part": part, "section": section})
        found = [s for s in parse_part_xml(xml, title, part, date) if s.section == section]
        if not found:
            raise LookupError(f"{title} CFR {section} not found on {date}")
        return found[0]

    def last_amended(self, title: int, part: str) -> dict[str, str]:
        data = json.loads(self._get(f"/api/versioner/v1/versions/title-{title}.json", {"part": part}))
        latest: dict[str, str] = {}
        for v in data.get("content_versions", []):
            if v.get("type") == "section" and v.get("identifier"):
                date = v.get("amendment_date") or v.get("date")
                if date and date > latest.get(v["identifier"], ""):
                    latest[v["identifier"]] = date
        return latest

    def search(self, query: str, title: int | None = None, per_page: int = 20) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": query, "per_page": per_page}
        if title is not None:
            params["hierarchy[title]"] = title
        data = json.loads(self._get("/api/search/v1/results", params, cacheable=False))
        return [
            {
                "title": r.get("hierarchy", {}).get("title"),
                "part": r.get("hierarchy", {}).get("part"),
                "section": r.get("hierarchy", {}).get("section"),
                "heading": (r.get("headings") or {}).get("section"),
                "excerpt": _clean(re.sub(r"<[^>]+>", "", r.get("full_text_excerpt") or "")),
                "starts_on": r.get("starts_on"),
            }
            for r in data.get("results", [])
        ]

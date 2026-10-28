"""Client for the public eCFR API (https://www.ecfr.gov/developers/documentation/api/v1).

Responses are cached on disk because CFR text for a given date never
changes, and the API is a shared public resource.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://www.ecfr.gov"
USER_AGENT = "regagents/0.1 (research prototype; +https://github.com/)"

# Paragraph designations nest (a) -> (1) -> (i) -> (A) -> (<i>1</i>) -> (<i>i</i>);
# eCFR marks levels 4 and 5 with <I> inside the parentheses.
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


_ROMAN_NEXT = {"i": "ii", "v": "vi", "x": "xi"}


def _level(token: str, stack: list[tuple[int, str]], next_token: str | None = None,
           italic: bool = False) -> int:
    if token.isdigit():
        return 4 if italic else 1
    if token.isupper():
        return 3
    deepest = stack[-1][0] if stack else -1
    if _ROMAN_RE.match(token) and (italic or deepest >= 1):
        if italic:
            return 5
        letter = next((t for lvl, t in stack if lvl == 0), "")
        follows_letter = len(token) == 1 and len(letter) == 1 and ord(token) == ord(letter) + 1
        # After (h)(5), "(i)" is either (h)(5)(i) or the next letter (i): it's
        # roman only if the following paragraph continues the numerals.
        if not follows_letter or next_token == _ROMAN_NEXT.get(token):
            return 2
    return 0


def _push(stack: list[tuple[int, str]], level: int, token: str) -> list[tuple[int, str]]:
    return [entry for entry in stack if entry[0] < level] + [(level, token)]


# eCFR often runs a heading and its first child(ren) into one <P>:
# "(b) Initial applicability. (1) The provisions ..." / "(h) Compliance—(1) Applicability."
_INLINE_RE = re.compile(r"^.{0,300}?(?:[.:]\s+|\u2014\s*)((?:\((?:[a-zA-Z]{1,5}|\d{1,3})\))+)\s")
_FIRST_CHILD = {0: "1", 1: "i", 2: "A", 3: "1", 4: "i"}


def _inline_children(text: str, level: int) -> list[str]:
    m = _INLINE_RE.search(text)
    if not m or not m.group(1).startswith(f"({_FIRST_CHILD.get(level)})"):
        return []
    return re.findall(r"\(([a-zA-Z]{1,5}|\d{1,3})\)", m.group(1))


def _leading_italic(p: ET.Element) -> bool:
    return bool((p.text or "").strip() == "(" and len(p) and p[0].tag == "I"
                and (p[0].tail or "").startswith(")"))


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
            raw = []
            for p in div.iter("P"):
                text = _clean("".join(p.itertext()))
                if not text:
                    continue
                labels, rest = [], text
                while (m := _LABEL_RE.match(rest)):
                    labels.append(m.group(1))
                    rest = rest[m.end():]
                raw.append((text, labels, _leading_italic(p)))
            paragraphs, stack = [], []
            for i, (text, labels, italic) in enumerate(raw):
                if labels:
                    following = next((lbls[0] for _, lbls, _ in raw[i + 1:] if lbls), None)
                    level = _level(labels[0], stack, labels[1] if len(labels) > 1 else following, italic)
                    stack = _push(stack, level, labels[0])
                    # a chain like "(c)(1)(i)" or an inline "(1)(i)" after a heading
                    # nests one level per designation
                    for token in labels[1:] + _inline_children(text, stack[-1][0]):
                        stack = _push(stack, stack[-1][0] + 1, token)
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
        self._titles: tuple[float, list[dict[str, Any]]] | None = None
        self._parsed: dict[tuple[int, str, str], list[Section]] = {}

    def _cache_file(self, path: str, params: dict[str, Any] | None) -> Path | None:
        key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / key if self.cache_dir else None

    def _get(self, path: str, params: dict[str, Any] | None = None, cacheable: bool = True) -> str:
        cached = self._cache_file(path, params)
        if cacheable and cached and cached.exists():
            return cached.read_text()
        resp = self.http.get(path, params=params)
        resp.raise_for_status()
        if cacheable and cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(resp.text)
        return resp.text

    def titles(self) -> list[dict[str, Any]]:
        # not cached on disk (up_to_date_as_of moves daily), but reused for an hour
        if self._titles is None or time.monotonic() - self._titles[0] > 3600:
            self._titles = (time.monotonic(), json.loads(self._get("/api/versioner/v1/titles.json", cacheable=False))["titles"])
        return self._titles[1]

    def current_date(self, title: int) -> str:
        for t in self.titles():
            if int(t["number"]) == int(title):
                return t["up_to_date_as_of"]
        raise ValueError(f"unknown CFR title {title}")

    def _parse_part(self, title: int, part: str, date: str) -> list[Section]:
        key = (int(title), str(part), date)
        if key not in self._parsed:
            xml = self._get(f"/api/versioner/v1/full/{date}/title-{title}.xml", {"part": str(part)})
            self._parsed = {key: parse_part_xml(xml, title, str(part), date)}  # keep only the latest part
        return self._parsed[key]

    def part(self, title: int, part: str, date: str | None = None) -> list[Section]:
        date = date or self.current_date(title)
        sections = copy.deepcopy(self._parse_part(title, part, date))
        amended = self.last_amended(title, part)
        for s in sections:
            s.last_amended = amended.get(s.section)
        return sections

    def section(self, title: int, section: str, date: str | None = None) -> Section:
        date = date or self.current_date(title)
        part = section.split(".")[0]
        part_cached = self._cache_file(f"/api/versioner/v1/full/{date}/title-{title}.xml", {"part": part})
        if (int(title), part, date) in self._parsed or (part_cached and part_cached.exists()):
            # same authoritative text for the same date, without one request per section
            found = [s for s in self._parse_part(title, part, date) if s.section == section]
        else:
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

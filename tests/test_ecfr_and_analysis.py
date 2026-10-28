from datetime import date

from ecfr_mock import transport

from regagents.analysis import extract_requirements, find_duplicates, find_outdated, split_sentences
from regagents.ecfr import EcfrApi, Paragraph, Section, parse_part_xml


def api():
    return EcfrApi(cache_dir="", transport=transport())


def test_parse_part_labels_and_metadata():
    sections = {s.section: s for s in api().part(99, "900")}
    assert list(sections) == ["900.1", "900.2", "900.3"]
    s = sections["900.2"]
    assert s.heading == "Notification and recordkeeping."
    assert [p.label for p in s.paragraphs] == ["(a)", "(b)(1)", "(b)(2)", "(b)(2)(i)", "(b)(2)(ii)", "(c)"]
    assert s.last_amended == "1997-03-01"
    assert s.citation == "99 CFR 900.2"
    assert "June 1, 1985" in s.paragraphs[-1].text  # inline <E> markup flattened


def test_letter_i_after_h_is_not_roman():
    xml = """<DIV8 N="§ 1.1" TYPE="SECTION"><HEAD>§ 1.1 X.</HEAD>
      <P>(h) Eighth.</P><P>(1) Sub.</P><P>(i) Ninth letter.</P><P>(1) Under i.</P><P>(i) Roman under (i)(1).</P></DIV8>"""
    labels = [p.label for p in parse_part_xml(xml, 1, "1")[0].paragraphs]
    assert labels == ["(h)", "(h)(1)", "(i)", "(i)(1)", "(i)(1)(i)"]


def test_section_lookup():
    s = api().section(99, "900.3")
    assert s.section == "900.3" and s.paragraphs[0].label == "(a)"


def test_sentence_split_keeps_abbreviations():
    text = "The U.S. Environmental Protection Agency shall receive reports. Records shall be kept, e.g. on paper."
    assert split_sentences(text) == ["The U.S. Environmental Protection Agency shall receive reports.",
                                      "Records shall be kept, e.g. on paper."]


def test_requirement_extraction_skips_non_obligations():
    reqs = extract_requirements(api().part(99, "900"))
    cites = [r.citation for r in reqs]
    assert "99 CFR 900.2(a)" in cites and "99 CFR 900.3(a)" in cites
    assert not any(c.startswith("99 CFR 900.1") for c in cites)  # "may waive" is not a requirement
    assert "99 CFR 900.2(b)(2)" not in cites  # "may be kept" is permissive, not an obligation


def test_duplicate_and_outdated_flags():
    reqs = extract_requirements(api().part(99, "900"))
    dupes = find_duplicates(reqs)
    assert dupes, "the two annual-report requirements should be flagged"
    top = dupes[0]
    assert {r.citation for r in top.requirements} == {"99 CFR 900.2(a)", "99 CFR 900.3(a)"}
    assert top.signals[0]["cosine"] >= 0.55

    outdated = {f.requirements[0].citation: f for f in find_outdated(reqs, today=date(2026, 10, 28))}
    assert "obsolete_technology" in outdated["99 CFR 900.2(b)(2)(ii)"].reason
    assert "past_deadline" in outdated["99 CFR 900.2(c)"].reason
    assert "defunct_agency" in outdated["99 CFR 900.3(b)"].reason
    assert "99 CFR 900.2(a)" not in outdated  # "not amended since 1997" alone is only a weak signal


def test_same_paragraph_sentences_are_not_duplicates():
    s = Section(1, "1", "1.1", "x", [Paragraph("(a)", "The operator shall submit a report. The operator shall submit a report.")])
    assert find_duplicates(extract_requirements([s])) == []

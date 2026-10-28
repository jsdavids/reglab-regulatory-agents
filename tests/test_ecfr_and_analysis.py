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
    assert top.signals[0]["min_cosine"] >= 0.55

    outdated = {f.requirements[0].citation: f for f in find_outdated(reqs, today=date(2026, 10, 28))}
    assert "obsolete_technology" in outdated["99 CFR 900.2(b)(2)(ii)"].reason
    assert "past_deadline" in outdated["99 CFR 900.2(c)"].reason
    assert "defunct_agency" in outdated["99 CFR 900.3(b)"].reason
    assert "99 CFR 900.2(a)" not in outdated  # "not amended since 1997" alone is only a weak signal


def test_same_paragraph_sentences_are_not_duplicates():
    s = Section(1, "1", "1.1", "x", [Paragraph("(a)", "The operator shall submit a report. The operator shall submit a report.")])
    assert find_duplicates(extract_requirements([s])) == []


def test_ambiguous_i_resolved_by_lookahead():
    # shape of 40 CFR 60.4375a: (h)(5)(i)-(iv) roman, then (h)(6), then the letter (i)
    xml = """<DIV8 N="§ 1.2" TYPE="SECTION"><HEAD>§ 1.2 X.</HEAD>
      <P>(h) Outage.</P><P>(5) Describe:</P><P>(i) When.</P><P>(ii) Why.</P><P>(6) Decision.</P>
      <P>(i) Force majeure.</P><P>(1) Claim.</P><P>(2) Notify.</P></DIV8>"""
    labels = [p.label for p in parse_part_xml(xml, 1, "1")[0].paragraphs]
    assert labels == ["(h)", "(h)(5)", "(h)(5)(i)", "(h)(5)(ii)", "(h)(6)", "(i)", "(i)(1)", "(i)(2)"]


def test_inline_first_child_after_heading():
    # shape of 40 CFR 63.1(b): heading and (1) share one paragraph
    xml = """<DIV8 N="§ 1.3" TYPE="SECTION"><HEAD>§ 1.3 X.</HEAD>
      <P>(b) <I>Initial applicability.</I> (1) The provisions apply to sources that:</P>
      <P>(i) Emit pollutants; and</P><P>(ii) Are subject to a standard.</P><P>(2) [Reserved]</P><P>(c) Next.</P></DIV8>"""
    labels = [p.label for p in parse_part_xml(xml, 1, "1")[0].paragraphs]
    assert labels == ["(b)(1)", "(b)(1)(i)", "(b)(1)(ii)", "(b)(2)", "(c)"]


def test_inline_chain_after_heading():
    # shape of 40 CFR 63.5(e): "(e) Heading. (1)(i) ..." then (ii)
    xml = """<DIV8 N="§ 1.4" TYPE="SECTION"><HEAD>§ 1.4 X.</HEAD>
      <P>(e) <I>Approval.</I> (1)(i) If the Administrator determines that:</P>
      <P>(ii) In addition, the determination will be based on:</P><P>(A) Cost.</P>
      <P>(<I>1</I>) Italic level.</P><P>(<I>i</I>) Italic roman.</P><P>(2) Next.</P>
      <P>(f) Compliance—(1) Applicability. Text.</P><P>(2) Methods.</P></DIV8>"""
    labels = [p.label for p in parse_part_xml(xml, 1, "1")[0].paragraphs]
    assert labels == ["(e)(1)(i)", "(e)(1)(ii)", "(e)(1)(ii)(A)", "(e)(1)(ii)(A)(1)", "(e)(1)(ii)(A)(1)(i)",
                      "(e)(2)", "(f)(1)", "(f)(2)"]


def test_repeated_boilerplate_becomes_one_group_without_chaining():
    boiler = "The owner or operator shall submit the notification postmarked 30 days before startup."
    sections = [Section(1, "1", f"1.{n}", "x", [Paragraph("(a)", boiler)]) for n in range(1, 6)]
    sections.append(Section(1, "1", "1.9", "x", [Paragraph("(a)", "Each operator shall keep records of opacity readings for five years.")]))
    groups = find_duplicates(extract_requirements(sections))
    assert len(groups) == 1 and len(groups[0].requirements) == 5  # not 10 pairs
    assert "5 near-identical reporting requirements across 5 sections" == groups[0].reason


def test_cross_references_distinguish_otherwise_identical_text():
    a = Section(1, "1", "1.1", "x", [Paragraph("(a)", "You must maintain records as specified in § 60.395a.")])
    b = Section(1, "1", "1.2", "x", [Paragraph("(a)", "You must maintain records as specified in § 60.5420.")])
    c = Section(1, "1", "1.3", "x", [Paragraph("(a)", "You must maintain records as specified in § 60.395a.")])
    groups = find_duplicates(extract_requirements([a, b, c]))
    assert [{r.section for r in g.requirements} for g in groups] == [{"1.1", "1.3"}]

# Labelling guide

`eval/to_label.jsonl` holds 96 items: 12 from each of 8 strata (`regagents eval
sample --title 40 --parts 60 63 --date 2026-10-26`, seed 0). Label them with

```bash
.venv/bin/regagents eval label eval/to_label.jsonl   # y/n, optionally followed by a note; s skips, q quits
.venv/bin/regagents eval score eval/to_label.jsonl
```

Each answer is saved as you go, so you can stop and resume. Open the eCFR link
when the sentence alone isn't enough, but judge the sentence as shown. Write a
note whenever you hesitate. The notes are where new outdated signals and
extractor fixes come from.

Decide these rules before you start, and don't change them halfway through. If
you do change one, relabel the items it affects.

## requirement: is this a reporting or recordkeeping requirement?

**Yes** if the sentence, on its own, obliges a regulated party (owner,
operator, "you", a manufacturer, a laboratory) to do one of these:

- report, submit, file or send information
- notify someone
- record information or keep a log
- keep or retain records or copies
- certify something in writing
- make records available on request

**No** for:

- operating duties ("operate and maintain the affected source …", "maintain the temperature above …")
- obligations on the Administrator or the EPA
- definitions
- applicability statements
- test methods with no reporting step
- permissive text ("may submit")
- "[Reserved]"

If the sentence splitter cut a sentence badly, label what's shown and note
`bad split`.

The strata tell you which regex half matched. `obligation_only` and
`reporting_only` are near-misses, and `neither` should be almost always No.
Any Yes in those three is a miss by the extractor, and each `neither` item
stands in for about 5,800 sentences.

## outdated: is the requirement outdated as of the item's `as_of` date?

**Yes** if the text needs editing to remove something obsolete:

- it requires or names an obsolete medium as a method: fax, telex, microfiche, magnetic tape, diskette
- it names an agency that no longer exists
- it sets a one-time deadline that has passed and no longer binds anyone
- it ties the obligation to a specific past year

**No** if:

- the old method appears only as one of several allowed options, and a current method is also offered (add the note `legacy option`)
- the deadline still applies to sources that become subject later (for example, "within 90 days after startup")
- the date only marks which sources are covered ("constructed after August 17, 1971")

When unsure, answer No and note `unsure`.

## duplicate: would one party bound by both report or record the same thing twice?

Both requirements are shown with their subparts. Every pair you see can bind
the same party: either both are in one subpart, or one is in the part's
General Provisions, which apply to every subpart.

**Yes** if a party subject to both would have to supply substantially the same
information (same content, same recipient, overlapping deadline or frequency),
or one text restates the other so that one could be deleted without losing any
obligation. A subpart repeating the General Provisions word for word is Yes.

**No** if:

- the texts differ in something that matters: a deadline, a frequency, a recipient, the pollutant or unit covered, the trigger, or the section they cite
- one modifies or replaces the other ("in lieu of § 63.10(b) …") — that's an override, not a duplicate
- they're the same template applied to different things, like performance test data and performance evaluation data

The report claims that each member of a group duplicates its seed. It makes no
claim about two members of the same group, so those pairs aren't in the sample.

## Agreement check

Ask a second person to label the same 20 items. Give them a copy of the file
with all labels cleared and only those 20 items kept, and this guide. Then run:

```bash
.venv/bin/regagents eval agree eval/to_label.jsonl eval/second_labeller.jsonl
```

Report Cohen's kappa next to the precision and recall figures. Talk through
the disagreements it lists. If one exposes an unclear rule, fix the rule here
and relabel.

## What the numbers cover

- **Requirement** precision and recall cover every sentence in the two parts.
- **Outdated** and **duplicate** numbers are conditional on extraction, since they only see sentences the extractor kept.
- **Duplicate** recall is relative to pairs within one scope with cosine ≥ 0.30.
- The report shows only the top 200 groups per type. The eval scores the uncapped detector.
- **12 items per stratum gives wide intervals.** To widen the sample without redoing any work, draw a larger one and bring your labels with you. Same seed, more per stratum, so it contains every item you've labelled, with weights recomputed:
  `regagents eval sample --title 40 --parts 60 63 --date 2026-10-26 --per-stratum 24 --carry eval/to_label.jsonl --out eval/to_label_24.jsonl`

"""Optional Claude judge that confirms or rejects heuristic candidates.

Enable with AGENTMESH_LLM=1 (see agentmesh.llm). The eval harness scores
the heuristic alone and heuristic + judge on the same labels, so you can
report whether the model actually helps.
"""

from __future__ import annotations

from typing import Any

from agentmesh.llm import ClaudeJSON

SYSTEM = (
    "You are a regulatory analyst reviewing the Code of Federal Regulations for burden "
    "reduction. Judge only from the quoted text. Be conservative: when the text does not "
    "clearly support the finding, answer false."
)

_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "boolean"}, "rationale": {"type": "string"}},
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


async def judge_duplicate(llm: ClaudeJSON, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Do these two reporting requirements ask the same regulated party to report or record "
        "substantially the same information, such that one could be consolidated into the other?\n\n"
        f"Requirement A ({a['citation']}):\n{a['text']}\n\n"
        f"Requirement B ({b['citation']}):\n{b['text']}"
    )
    return await llm.complete(system=SYSTEM, prompt=prompt, schema=_SCHEMA, max_tokens=2000)


async def judge_outdated(llm: ClaudeJSON, req: dict[str, Any], signals: list[dict[str, Any]]) -> dict[str, Any]:
    hints = "\n".join(f"- {s['kind']}: {s['match']} ({s['note']})" for s in signals) or "- none"
    prompt = (
        "Is this reporting requirement outdated: it relies on obsolete technology, a defunct "
        "agency, or a deadline or period that has passed?\n\n"
        f"Requirement ({req['citation']}):\n{req['text']}\n\nAutomated signals:\n{hints}"
    )
    return await llm.complete(system=SYSTEM, prompt=prompt, schema=_SCHEMA, max_tokens=2000)

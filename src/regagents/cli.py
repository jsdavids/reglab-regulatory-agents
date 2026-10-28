"""regagents command line.

    regagents review --title 40 --parts 60          run the agents (needs `agentmesh up mesh.yaml`)
    regagents eval sample --title 40 --parts 60 63  write eval/to_label.jsonl
    regagents eval score eval/labels.jsonl [--judge]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agentmesh import MeshClient, MeshConfig, TokenIssuer
from regagents import eval as ev
from regagents.orchestrator import review, to_markdown


def _mesh(config_path: str) -> MeshClient:
    config = MeshConfig.load(config_path)
    token = os.environ.get("AGENTMESH_TOKEN") or TokenIssuer(
        config.secret, config.gateway.issuer, config.gateway.audience).mint("orchestrator")
    return MeshClient(config.gateway.url, token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regagents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_review = sub.add_parser("review", help="flag duplicate/outdated reporting requirements")
    p_review.add_argument("--config", default="mesh.yaml")
    p_review.add_argument("--title", type=int, required=True)
    p_review.add_argument("--parts", nargs="+", required=True)
    p_review.add_argument("--date", help="YYYY-MM-DD (default: current)")
    p_review.add_argument("--threshold", type=float, default=0.55)
    p_review.add_argument("--out", type=Path, help="write report JSON and .md here")

    p_eval = sub.add_parser("eval", help="hand-label evaluation")
    eval_sub = p_eval.add_subparsers(dest="action", required=True)
    p_sample = eval_sub.add_parser("sample")
    p_sample.add_argument("--title", type=int, required=True)
    p_sample.add_argument("--parts", nargs="+", required=True)
    p_sample.add_argument("--date")
    p_sample.add_argument("--per-stratum", type=int, default=25)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--out", type=Path, default=Path("eval/to_label.jsonl"))
    p_score = eval_sub.add_parser("score")
    p_score.add_argument("labels", type=Path)
    p_score.add_argument("--threshold", type=float, default=0.55)
    p_score.add_argument("--judge", action="store_true", help="confirm heuristic positives with Claude")

    args = parser.parse_args(argv)
    if args.cmd == "review":
        report = asyncio.run(review(_mesh(args.config), args.title, args.parts, args.date, args.threshold))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
            args.out.with_suffix(".md").write_text(to_markdown(report))
            print(f"wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")
        else:
            print(to_markdown(report))
    elif args.action == "sample":
        n = ev.sample_command(args.title, args.parts, args.out, args.per_stratum, args.seed, args.date)
        print(f"wrote {n} items to {args.out}; set \"label\" to true/false on each line, then run "
              f"`regagents eval score {args.out}`")
    else:
        items = ev.load_items(args.labels)
        predictions = asyncio.run(ev.judged_predictions(items, args.threshold)) if args.judge else None
        print(json.dumps(ev.score(items, args.threshold, predictions), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

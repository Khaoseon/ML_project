"""Print the 90 open-ended evaluation questions grouped by domain.

    python scripts/list_questions.py                      # all domains
    python scripts/list_questions.py --domain disinformation
    python scripts/list_questions.py --markdown > questions.md

The file is read from $MEM_DATA/eval/open_ended/eval_input_broad90_photo.json
(default: ../vlm-em/data). See the README for the download command.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_DIR.parent / "vlm-em" / "data"
QUESTIONS_FILE = Path("eval") / "open_ended" / "eval_input_broad90_photo.json"


def load_questions(data_dir: Path) -> dict[str, list[dict]]:
    """Return {domain: [question, ...]} in file order."""
    path = data_dir / QUESTIONS_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found; download it first (see README)")
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for question in json.loads(path.read_text()):
        by_domain[question["domain"]].append(question)
    return dict(by_domain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("MEM_DATA", DEFAULT_DATA)))
    parser.add_argument("--domain", help="only print this domain")
    parser.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = parser.parse_args()

    by_domain = load_questions(args.data)
    if args.domain:
        if args.domain not in by_domain:
            raise SystemExit(f"unknown domain {args.domain!r}; choose from {sorted(by_domain)}")
        by_domain = {args.domain: by_domain[args.domain]}

    for domain, questions in by_domain.items():
        if args.markdown:
            print(f"### `{domain}`\n\n| id | question |\n|---|---|")
            for q in questions:
                print(f"| `{q['id']}` | {q['question_text']} |")
            print()
        else:
            print(f"== {domain} ({len(questions)})")
            for q in questions:
                print(f"  {q['id']:<30} {q['question_text']}")


if __name__ == "__main__":
    main()

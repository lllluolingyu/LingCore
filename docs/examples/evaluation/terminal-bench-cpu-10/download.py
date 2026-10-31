"""Export just this suite's tasks at their bundled publisher digests."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor", help="Existing Harbor 0.25.0 executable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    example = Path(__file__).resolve().parent
    repository = example.parents[3]
    suite = yaml.safe_load((example / "suite.yaml").read_text())
    release = json.loads(
        (
            repository / "lingcore/evaluation/harbor/releases/terminal-bench-4.0.0.json"
        ).read_text()
    )
    if suite["source"]["dataset"] != release["dataset"]:
        parser.error("suite dataset differs from the bundled release")
    harbor = (
        [args.harbor] if args.harbor else ["uvx", "--python", "3.12", "harbor@0.25.0"]
    )
    for task in suite["source"]["selection"]["tasks"]:
        command = [
            *harbor,
            "download",
            f"terminal-bench/{task}@{release['tasks'][task]}",
            "--export",
            "-o",
            str(example / suite["source"]["path"]),
        ]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

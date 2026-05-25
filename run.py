"""Run a repository script with the local module paths configured."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATHS = [
    ROOT / "src",
    ROOT / "scripts" / "analysis",
    ROOT / "scripts" / "siena",
    ROOT / "scripts" / "preprocessing",
    ROOT / "scripts" / "training",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a script from this repository.")
    parser.add_argument("script", help="Script path relative to the repository root.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = (ROOT / args.script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    if ROOT not in script.parents:
        raise ValueError(f"Script must live under {ROOT}")

    for path in reversed(PATHS):
        sys.path.insert(0, str(path))

    sys.argv = [str(script), *args.script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

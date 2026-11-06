"""Run the frozen algorithm on a new output snapshot, preserving released results."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--actions", nargs="+", default=["freeze", "baseline", "search", "select", "test"],
                        choices=["freeze", "baseline", "search", "select", "test", "smoke", "environment"])
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Use an output directory outside the released repository.")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if out.exists() and not args.resume:
        parser.error("Output exists; use --resume to continue without replacing files.")
    if not out.exists():
        out.mkdir(parents=True)
        for name in ["src", "tests", "data", "reference", "environment", "tools", "notebooks"]:
            shutil.copytree(ROOT / name, out / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for action in args.actions:
        print("Running", action, "in", out, flush=True)
        subprocess.run([sys.executable, str(out / "src/engine.py"), action,
                        "--root", str(out), "--workers", str(args.workers)], env=env, check=True)


if __name__ == "__main__":
    main()

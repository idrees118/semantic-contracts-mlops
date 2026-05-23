#!/usr/bin/env python3
"""
main.py
========
Single entry point for the semantic mutation v2 experiment.

Typical usage
-------------
# 1. Prepare clean datasets (run once):
    python main.py --prepare

# 2. Run the full experiment:
    python main.py --experiment

# 3. Run held-out evaluation (after experiment):
    python main.py --held-out

# 4. Run everything in sequence:
    python main.py --all

# 5. Verify installation with smoke tests:
    python main.py --test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_prepare(args):
    print("\n" + "=" * 60)
    print("STEP 1 — Preparing datasets")
    print("=" * 60)
    cmd = [sys.executable, str(ROOT / "scripts" / "prepare_datasets.py")]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_experiment(args):
    print("\n" + "=" * 60)
    print("STEP 2 — Running experiment")
    print("=" * 60)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
        "--walmart", str(ROOT / "data/processed/walmart_clean.csv"),
        "--stock",   str(ROOT / "data/processed/stock_clean.csv"),
        "--power",   str(ROOT / "data/processed/power_clean.csv"),
        "--seeds",   *[str(s) for s in args.seeds],
        "--out",     str(ROOT / "experiments/results"),
    ]
    if args.quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_held_out(args):
    print("\n" + "=" * 60)
    print("STEP 3 — Held-out evaluation")
    print("=" * 60)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "held_out_evaluation.py"),
        "--results", str(ROOT / "experiments/results/per_mutation_detection_rates.csv"),
        "--out",     str(ROOT / "experiments/results/held_out_evaluation.csv"),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_tests(args):
    print("\n" + "=" * 60)
    print("SMOKE TESTS")
    print("=" * 60)
    try:
        import pytest
    except ImportError:
        print("[ERROR] pytest not installed. Run: pip install pytest")
        return 1
    return pytest.main([str(ROOT / "tests"), "-v", "--tb=short"])


def parse_args():
    p = argparse.ArgumentParser(
        description="Semantic Mutation v2 — main entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",    action="store_true", help="Prepare/clean raw datasets")
    group.add_argument("--experiment", action="store_true", help="Run the full experiment")
    group.add_argument("--held-out",   action="store_true", help="Run held-out evaluation")
    group.add_argument("--test",       action="store_true", help="Run smoke tests")
    group.add_argument("--all",        action="store_true", help="Run prepare → experiment → held-out")

    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 123, 456, 789],
                   help="Random seeds (default: 42 123 456 789)")
    p.add_argument("--quiet",  action="store_true", help="Suppress per-mutation verbose output")
    return p.parse_args()


def main():
    args = parse_args()

    if args.prepare:
        sys.exit(run_prepare(args))

    elif args.experiment:
        sys.exit(run_experiment(args))

    elif args.held_out:
        sys.exit(run_held_out(args))

    elif args.test:
        sys.exit(run_tests(args))

    elif args.all:
        rc = run_prepare(args)
        if rc != 0:
            print(f"\n[ABORT] prepare_datasets failed with exit code {rc}")
            sys.exit(rc)
        rc = run_experiment(args)
        if rc != 0:
            print(f"\n[ABORT] run_experiment failed with exit code {rc}")
            sys.exit(rc)
        rc = run_held_out(args)
        sys.exit(rc)


if __name__ == "__main__":
    main()

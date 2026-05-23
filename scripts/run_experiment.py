#!/usr/bin/env python3
"""
scripts/run_experiment.py
==========================
Main entry point for the v2 semantic mutation testing experiment.

Usage
-----
    python scripts/run_experiment.py \
        --walmart  data/processed/walmart_clean.csv \
        --stock    data/processed/stock_clean.csv \
        --power    data/processed/power_clean.csv \
        --seeds    42 123 456 789 \
        --out      experiments/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.runner import run_full_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run semantic mutation v2 experiment.")
    p.add_argument("--walmart", type=Path, default=ROOT / "data/processed/walmart_clean.csv")
    p.add_argument("--stock",   type=Path, default=ROOT / "data/processed/stock_clean.csv")
    p.add_argument("--power",   type=Path, default=ROOT / "data/processed/power_clean.csv")
    p.add_argument("--seeds",   type=int, nargs="+", default=[42, 123, 456, 789])
    p.add_argument("--out",     type=Path, default=ROOT / "experiments/results")
    p.add_argument("--quiet",   action="store_true")
    return p.parse_args()


def load_dataset(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Dataset not found: {path}  (skipping {name})")
        return None
    df = pd.read_csv(path)
    print(f"[OK]   Loaded {name}: {len(df):,} rows × {len(df.columns)} cols  ({path.name})")
    return df


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for name, path in [("walmart", args.walmart), ("stock", args.stock), ("power", args.power)]:
        df = load_dataset(path, name)
        if df is not None:
            datasets[name] = df

    if not datasets:
        print("[ERROR] No datasets loaded. Exiting.")
        sys.exit(1)

    print(f"\nRunning experiment with seeds={args.seeds}, {len(datasets)} datasets ...")
    results_df, summary = run_full_experiment(
        datasets, seeds=args.seeds, verbose=not args.quiet
    )

    # ---- Save raw results ----
    raw_path = args.out / "raw_results.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] raw_results.csv  ({len(results_df)} rows)")

    # ---- Save mutated-only summary ----
    mut_df = results_df[results_df["is_mutated"]].copy()
    mut_df.to_csv(args.out / "mutation_results.csv", index=False)

    # ---- Save per-mutation detection rates across seeds ----
    agg = (
        mut_df.groupby("mutation_name")
        .agg(
            dataset        = ("dataset",           "first"),
            category       = ("category",          "first"),
            sem_dr         = ("sem_detected",       "mean"),
            ens_dr         = ("ensemble_detected",  "mean"),
            ks_dr          = ("ks_detected",        "mean"),
            ge_dr          = ("ge_detected",        "mean"),   # GE column
            sem_match_rate = ("sem_semantic_match", "mean"),
            n_seeds        = ("seed",               "count"),
        )
        .reset_index()
        .sort_values(["dataset", "category", "mutation_name"])
    )
    agg.to_csv(args.out / "per_mutation_detection_rates.csv", index=False)
    print(f"[SAVED] per_mutation_detection_rates.csv")

    # ---- Save detection summary ----
    summary_rows = []
    for val_name, val_summary in summary["validators"].items():
        fpr = summary["fpr"].get(val_name, float("nan"))
        summary_rows.append({
            "Validator":          val_name,
            "Mean DR":            f"{val_summary['mean_dr']*100:.1f}%",
            "Std DR":             f"±{val_summary['std_dr']*100:.1f}%",
            "Pooled DR (95% CI)": val_summary["pooled_ci"],
            "FPR":                f"{fpr:.3f}",
        })
    pd.DataFrame(summary_rows).to_csv(args.out / "detection_summary.csv", index=False)
    print(f"[SAVED] detection_summary.csv")

    # ---- Save McNemar results ----
    mcn_rows = [
        {
            "Comparison":         comparison,
            "p-value":            f"{r['p_value']:.4f}",
            "Significant":        "Yes" if r["significant"] else "No",
        }
        for comparison, r in summary["mcnemar"].items()
    ]
    pd.DataFrame(mcn_rows).to_csv(args.out / "mcnemar_results.csv", index=False)
    print(f"[SAVED] mcnemar_results.csv")

    # ---- Save semantic precision ----
    sp = summary["semantic_precision"]
    sp_rows = sp["per_category"] + [{
        "category":         "OVERALL",
        "detected":         sp["total_detected"],
        "semantic_matches": sp["total_semantic_matches"],
        "precision":        sp["overall_precision"],
    }]
    pd.DataFrame(sp_rows).to_csv(args.out / "semantic_precision.csv", index=False)
    print(f"[SAVED] semantic_precision.csv")

    # ---- Save per-category breakdown ----
    cat_rows = [{"category": cat, **r} for cat, r in sorted(summary["per_category"].items())]
    pd.DataFrame(cat_rows).to_csv(args.out / "per_category_breakdown.csv", index=False)
    print(f"[SAVED] per_category_breakdown.csv")

    print(f"\nAll outputs saved to: {args.out}/")
    return results_df, summary


if __name__ == "__main__":
    main()
"""
scripts/held_out_evaluation.py  (v2 — with GE baseline column)
================================
See docstring in original for full partition rationale.
This version adds ge_dr throughout to match the updated results CSV schema.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validators.semantic_validator import wilson_ci


# ---------------------------------------------------------------------------
# Partition definition  (unchanged from fixed version)
# ---------------------------------------------------------------------------

DEVELOPMENT_SET = frozenset([
    "walmart__shift_dates_-7",
    "walmart__shift_dates_7",
    "walmart__scale_sales_0.9",
    "walmart__toggle_holiday",
    "walmart__missing_recent_10pct",
    "walmart__wrong_aggregation_x7",
    "walmart__duplicate_rows_5pct",
    "stock__column_swap_open_close",
    "stock__shift_price_0.5",
    "stock__unit_conversion_x10",
    "power__scale_active_1.1",
    "power__shuffle_within_day",
    "power__column_swap_voltage_intensity",
    "power__missing_6hour_segment",
    "power__data_repetition_5x",
    "power__inject_spikes_100_3.0",
    "power__extreme_outliers_50",
    "power__shift_by_1week",
    "power__sign_flip_reactive",
])

HELD_OUT_SET = frozenset([
    "walmart__scale_temperature_2x",
    "walmart__missing_stores_5",
    "stock__noise_0.01",
    "stock__drop_5pct",
    "stock__reverse_series",
    "power__scale_voltage_0.5",
    "power__constant_segment_2h",
    "stock__timestamp_lag_1day",
    "stock__zero_volume_30pct",
])

assert len(DEVELOPMENT_SET) == 19
assert len(HELD_OUT_SET)    == 9
assert DEVELOPMENT_SET.isdisjoint(HELD_OUT_SET)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reconstruct_counts(df: pd.DataFrame, col: str) -> tuple[int, int]:
    raw_col = col.replace("_dr", "_detected")
    if raw_col in df.columns:
        return int(df[raw_col].sum()), int(df["n_seeds"].sum())
    reconstructed = df[col] * df["n_seeds"]
    rounded       = reconstructed.round().astype(int)
    error         = (reconstructed - rounded).abs().max()
    if error > 0.01:
        print(f"[WARN] Reconstruction error {error:.4f} for column '{col}'")
    return int(rounded.sum()), int(df["n_seeds"].sum())


def _subset_stats(df: pd.DataFrame, subset: frozenset, label: str) -> dict:
    sub = df[df["mutation_name"].isin(subset)].copy()
    if sub.empty:
        return {
            "label": label, "n_mutations": 0,
            "sem_dr": 0.0, "sem_ci": "N/A",
            "ens_dr": 0.0, "ks_dr": 0.0, "ge_dr": 0.0,
            "sem_detected": 0, "sem_total": 0, "per_category": [],
        }

    sem_detected, sem_total = _reconstruct_counts(sub, "sem_dr")
    ens_detected, _         = _reconstruct_counts(sub, "ens_dr")
    ks_detected,  _         = _reconstruct_counts(sub, "ks_dr")
    # ge_dr is optional — older result files may not have it
    ge_detected = 0
    if "ge_dr" in sub.columns:
        ge_detected, _ = _reconstruct_counts(sub, "ge_dr")

    lo, hi, ci_str = wilson_ci(sem_detected, sem_total)

    cat_rows = []
    for cat, g in sub.groupby("category"):
        n_d, n_t = _reconstruct_counts(g, "sem_dr")
        _, _, cat_ci = wilson_ci(n_d, n_t)
        cat_rows.append({"category": cat, "detected": n_d, "total": n_t, "rate_ci": cat_ci})

    return {
        "label":        label,
        "n_mutations":  len(sub),
        "sem_detected": sem_detected,
        "sem_total":    sem_total,
        "sem_dr":       sem_detected / sem_total if sem_total > 0 else 0.0,
        "sem_ci":       ci_str,
        "ens_detected": ens_detected,
        "ens_dr":       ens_detected / sem_total if sem_total > 0 else 0.0,
        "ks_detected":  ks_detected,
        "ks_dr":        ks_detected  / sem_total if sem_total > 0 else 0.0,
        "ge_detected":  ge_detected,
        "ge_dr":        ge_detected  / sem_total if sem_total > 0 else 0.0,
        "per_category": cat_rows,
    }


def _two_proportion_ztest(s1: dict, s2: dict) -> tuple[float, float]:
    n1, x1 = s1["sem_total"], s1["sem_detected"]
    n2, x2 = s2["sem_total"], s2["sem_detected"]
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    denom  = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return float("nan"), float("nan")
    z = ((x1/n1) - (x2/n2)) / denom
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _per_mutation_table(df: pd.DataFrame, subset: frozenset) -> pd.DataFrame:
    cols = ["mutation_name", "category", "sem_dr", "ens_dr", "ks_dr", "n_seeds"]
    if "ge_dr" in df.columns:
        cols.insert(5, "ge_dr")
    return (
        df[df["mutation_name"].isin(subset)][cols]
        .sort_values(["category", "mutation_name"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=ROOT / "experiments/results/per_mutation_detection_rates.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/results/held_out_evaluation.csv")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"[ERROR] Results file not found: {args.results}")
        sys.exit(1)

    df = pd.read_csv(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    has_ge = "ge_dr" in df.columns

    present      = set(df["mutation_name"])
    dev_present  = DEVELOPMENT_SET & present
    held_present = HELD_OUT_SET     & present
    unassigned   = present - DEVELOPMENT_SET - HELD_OUT_SET

    dev_stats  = _subset_stats(df, dev_present,  "Development Set")
    held_stats = _subset_stats(df, held_present, "Held-Out Set")
    all_stats  = _subset_stats(df, present,      "All Mutations (pooled)")

    # ── Console report ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("HELD-OUT EVALUATION REPORT")
    print("=" * 72)
    print(f"\nPartition sizes:")
    print(f"  Development set  : {len(dev_present)}/{len(DEVELOPMENT_SET)} mutations present")
    print(f"  Held-out set     : {len(held_present)}/{len(HELD_OUT_SET)} mutations present")
    if unassigned:
        print(f"  Unassigned       : {sorted(unassigned)}")

    ge_col = "GE DR" if has_ge else ""
    header = f"{'Set':<25} {'Sem DR (95% CI)':<38} {'Ens DR':>8} {'KS DR':>7}"
    if has_ge:
        header += f" {'GE DR':>7}"
    print(f"\n{header}")
    print("-" * (72 + (8 if has_ge else 0)))
    for s in [dev_stats, held_stats, all_stats]:
        line = (f"{s['label']:<25} {s['sem_ci']:<38} "
                f"{s['ens_dr']*100:>7.1f}% {s['ks_dr']*100:>6.1f}%")
        if has_ge:
            line += f" {s['ge_dr']*100:>6.1f}%"
        print(line)

    print(f"\nDevelopment set — per-category breakdown (Semantic Validator):")
    for row in dev_stats.get("per_category", []):
        print(f"  {row['category']:<15} {row['detected']}/{row['total']}  {row['rate_ci']}")

    print(f"\nHeld-out set — per-mutation results:")
    if held_present:
        held_table = _per_mutation_table(df, held_present)
        for _, r in held_table.iterrows():
            sem_pct = f"{r['sem_dr']*100:.0f}%"
            ens_pct = f"{r['ens_dr']*100:.0f}%"
            ks_pct  = f"{r['ks_dr']*100:.0f}%"
            ge_str  = f"  ge={r['ge_dr']*100:.0f}%" if has_ge else ""
            note = ""
            if r["mutation_name"] == "stock__noise_0.01":
                note = "  ← documented hard limit"
            if r["mutation_name"] == "stock__drop_5pct" and r["sem_dr"] > 0:
                note = "  ← detected by adjacent-category detector"
            print(f"  {r['mutation_name']:<47} sem={sem_pct:>4}  "
                  f"ens={ens_pct:>4}  ks={ks_pct:>4}{ge_str}{note}")

    # ── Generalisation gap + significance test ──────────────────────────────
    gap    = dev_stats["sem_dr"] - held_stats["sem_dr"]
    z, p   = _two_proportion_ztest(dev_stats, held_stats)

    print(f"\nGeneralisation gap  (dev DR - held-out DR): {gap*100:+.1f}%")
    print(f"Two-proportion z-test: z={z:.3f},  p={p:.4f}")
    if p > 0.05:
        print("  ✓ No statistically significant difference (dev vs held-out).")
    else:
        print("  ~ Significant gap — discuss in limitations.")
    if gap < 0:
        print(f"  Note: negative gap reflects partition composition — see Section VII.")
    elif gap < 0.10:
        print(f"  ✓ Gap < 10pp — detectors generalise to unseen mutation types.")

    # ── Save outputs ────────────────────────────────────────────────────────
    out_rows = []
    for s in [dev_stats, held_stats, all_stats]:
        row = {
            "set":          s["label"],
            "n_mutations":  s["n_mutations"],
            "sem_detected": s["sem_detected"],
            "sem_total":    s["sem_total"],
            "sem_dr":       f"{s['sem_dr']:.4f}",
            "sem_ci":       s["sem_ci"],
            "ens_dr":       f"{s['ens_dr']:.4f}",
            "ks_dr":        f"{s['ks_dr']:.4f}",
        }
        if has_ge:
            row["ge_dr"] = f"{s['ge_dr']:.4f}"
        if s["label"] == "Held-Out Set":
            row["z_stat"]  = f"{z:.3f}"
            row["p_value"] = f"{p:.4f}"
        out_rows.append(row)

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\n[SAVED] {args.out}")

    df["partition"] = df["mutation_name"].apply(
        lambda x: "development" if x in DEVELOPMENT_SET
        else ("held_out" if x in HELD_OUT_SET else "unassigned")
    )
    part_cols = ["mutation_name", "dataset", "category", "partition", "sem_dr", "ens_dr", "ks_dr"]
    if has_ge:
        part_cols.append("ge_dr")
    partition_path = args.out.parent / "mutation_partition.csv"
    df[part_cols].to_csv(partition_path, index=False)
    print(f"[SAVED] {partition_path}")

    return dev_stats, held_stats


if __name__ == "__main__":
    main()
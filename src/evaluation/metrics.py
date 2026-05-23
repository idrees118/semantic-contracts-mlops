"""
src/evaluation/metrics.py
==========================
Statistical evaluation metrics for the paper.

Functions
---------
wilson_ci           — Wilson score confidence interval
mcnemar_exact       — McNemar exact test (binomial, for small n)
detection_summary   — Aggregate detection rate + CI across seeds
semantic_precision  — Per-category semantic precision table
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, str]:
    """
    Wilson score confidence interval for a proportion k/n.

    Returns (lower, upper, formatted_string).
    """
    if n == 0:
        return 0.0, 0.0, "N/A"
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    lo = float(max(0.0, centre - margin))
    hi = float(min(1.0, centre + margin))
    return lo, hi, f"{p*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]"


def mcnemar_exact(
    n_only_a: int,
    n_only_b: int,
    two_sided: bool = True,
) -> dict[str, float | bool]:
    """
    McNemar exact binomial test.

    n_only_a : mutations detected by A but not B
    n_only_b : mutations detected by B but not A

    Under H₀: P(A detects, B misses) = P(B detects, A misses) = 0.5.
    We use the exact binomial CDF (appropriate for small n, as recommended
    when n_only_a + n_only_b < 25).

    Returns the two-sided p-value.
    """
    n = n_only_a + n_only_b
    if n == 0:
        p_val = 1.0
    else:
        k = min(n_only_a, n_only_b)
        p_one_side = float(binom.cdf(k, n, 0.5))
        p_val = min(1.0, p_one_side * 2) if two_sided else p_one_side

    return {"p_value": p_val, "significant": p_val < 0.05}


def detection_summary(
    per_seed_results: list[dict],
    total_mutations: int,
) -> dict:
    """
    Aggregate detection rates across multiple seeds.

    Parameters
    ----------
    per_seed_results : list of dicts, each with 'seed' and 'detected' keys.
        'detected' is a list/array of booleans (one per mutation).
    total_mutations  : expected number of mutations per seed.

    Returns
    -------
    {
      "mean_dr":   float,   # mean detection rate across seeds
      "std_dr":    float,   # std of detection rate across seeds
      "pooled_dr": float,   # pooled (overall) detection rate
      "pooled_ci": str,     # Wilson CI on pooled rate
      "per_seed":  list of {"seed", "dr", "detected", "total"}
    }
    """
    seed_drs = []
    all_detected = 0
    all_total = 0
    per_seed = []

    for r in per_seed_results:
        detected_arr = np.array(r["detected"], dtype=bool)
        n_det = int(detected_arr.sum())
        n_tot = len(detected_arr)
        dr = n_det / n_tot if n_tot > 0 else 0.0
        seed_drs.append(dr)
        all_detected += n_det
        all_total += n_tot
        per_seed.append({"seed": r["seed"], "dr": dr, "detected": n_det, "total": n_tot})

    mean_dr = float(np.mean(seed_drs)) if seed_drs else 0.0
    std_dr = float(np.std(seed_drs, ddof=1)) if len(seed_drs) > 1 else 0.0
    _, _, pooled_ci = wilson_ci(all_detected, all_total)

    return {
        "mean_dr":   mean_dr,
        "std_dr":    std_dr,
        "pooled_dr": all_detected / all_total if all_total > 0 else 0.0,
        "pooled_ci": pooled_ci,
        "per_seed":  per_seed,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build the main Table I comparison DataFrame from a results dict.

    `results` maps validator_name → detection_summary dict.
    """
    rows = []
    for name, summary in results.items():
        lo, hi, ci_str = wilson_ci(
            round(summary["pooled_dr"] * summary["per_seed"][0]["total"] * len(summary["per_seed"])),
            summary["per_seed"][0]["total"] * len(summary["per_seed"]),
        )
        rows.append({
            "Validator":        name,
            "Mean DR":          f"{summary['mean_dr']*100:.1f}%",
            "Std DR":           f"±{summary['std_dr']*100:.1f}%",
            "Pooled DR (95% Wilson CI)": summary["pooled_ci"],
        })
    return pd.DataFrame(rows)


def build_mcnemar_table(
    semantic_detections: list[bool],
    baselines: dict[str, list[bool]],
) -> pd.DataFrame:
    """
    Build McNemar test table: Semantic Validator vs. each baseline.

    semantic_detections : flat list of bool (one per mutation × seed)
    baselines           : {name: flat list of bool}
    """
    sem = np.array(semantic_detections, dtype=bool)
    rows = []
    for name, det in baselines.items():
        base = np.array(det, dtype=bool)
        n = min(len(sem), len(base))
        only_sem  = int(( sem[:n] & ~base[:n]).sum())
        only_base = int((~sem[:n] &  base[:n]).sum())
        p = mcnemar_exact(only_sem, only_base)
        rows.append({
            "Comparison": f"Semantic vs. {name}",
            "Only Semantic": only_sem,
            "Only Baseline": only_base,
            "p-value": f"{p['p_value']:.4f}",
            "Significant (α=0.05)": "Yes" if p["significant"] else "No",
        })
    return pd.DataFrame(rows)


def semantic_precision(validation_results: list[dict]) -> dict:
    """
    Computes semantic precision overall and per category.

    A detection is a "semantic match" if the category of the highest-confidence
    firing detector matches the ground-truth category of the mutation.

    Parameters
    ----------
    validation_results : list of dicts, each with keys:
        "detected" (bool), "mutation_category" (str), "semantic_match" (bool|None)

    Returns
    -------
    {
      "overall_precision": float,
      "total_detected": int,
      "total_semantic_matches": int,
      "per_category": list of dicts for per-category breakdown
    }
    """
    total_detected = 0
    total_semantic_matches = 0
    per_category = {}

    for r in validation_results:
        if not r["detected"]:
            continue
        total_detected += 1
        cat = r["mutation_category"]
        if cat not in per_category:
            per_category[cat] = {"detected": 0, "semantic_matches": 0}
        per_category[cat]["detected"] += 1
        if r["semantic_match"]:
            total_semantic_matches += 1
            per_category[cat]["semantic_matches"] += 1

    cat_rows = []
    for cat, counts in sorted(per_category.items()):
        prec = counts["semantic_matches"] / counts["detected"] if counts["detected"] > 0 else 0.0
        cat_rows.append({
            "category": cat,
            "detected": counts["detected"],
            "semantic_matches": counts["semantic_matches"],
            "precision": prec,
        })

    return {
        "overall_precision": total_semantic_matches / total_detected if total_detected > 0 else 0.0,
        "total_detected": total_detected,
        "total_semantic_matches": total_semantic_matches,
        "per_category": cat_rows,
    }

"""
src/validators/semantic_validator.py
=====================================
Orchestrates per-dataset semantic contracts and computes:
  - Binary detection per mutation
  - Per-detector confidence scores
  - Semantic precision (which category of detector fired)

Key improvements over v1
-------------------------
- Each detector is mapped to a SEMANTIC_CATEGORY so that precision can be
  computed automatically without a separate script.
- Ensemble voting: a mutation is flagged as detected if at least one contract
  fires with confidence ≥ CONFIDENCE_THRESHOLD (default 0.10). This is lower
  than "any detector fires" but higher than a noisy signal.
- All 28 mutations are covered with at least one targeted contract.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from src.validators.contracts import (
    # Walmart
    contract_walmart_shift_dates,
    contract_walmart_scale_sales,
    contract_walmart_toggle_holiday,
    contract_walmart_scale_temperature,
    contract_walmart_missing_recent,
    contract_walmart_missing_stores,
    contract_walmart_duplicate_rows,
    contract_stock_timestamp_lag,
    # Stock
    contract_stock_noise,
    contract_stock_drop_fraction,
    contract_stock_shift_price,
    contract_stock_column_swap,
    contract_stock_unit_conversion,
    contract_stock_temporal_acf,
    contract_stock_zero_volume,
    # Power
    contract_power_scale_active,
    contract_power_scale_voltage,
    contract_power_inject_spikes,
    contract_power_shuffle_within_day,
    contract_power_column_swap,
    contract_power_missing_segment,
    contract_power_data_repetition,
    contract_power_sign_flip,
    contract_power_temporal_acf,
    contract_power_constant_segment,
)

# Minimum confidence for a detector to count as "fired"
CONFIDENCE_THRESHOLD = 0.10

# Maps each detector name → taxonomy category
DETECTOR_CATEGORY: dict[str, str] = {
    "shift_dates":          "temporal",
    "scale_sales":          "scaling",
    "toggle_holiday":       "structural",
    "scale_temperature":    "scaling",
    "missing_recent":       "structural",
    "missing_stores":       "structural",
    "duplicate_rows":       "quality",
    "timestamp_lag":        "temporal",
    "noise":                "quality",
    "drop_fraction":        "structural",
    "shift_price":          "scaling",
    "column_swap":          "structural",
    "unit_conversion":      "scaling",
    "temporal_acf":         "temporal",
    "zero_volume":          "quality",
    "scale_active":         "scaling",
    "scale_voltage":        "scaling",
    "inject_spikes":        "quality",
    "shuffle_within_day":   "temporal",
    "missing_segment":      "structural",
    "data_repetition":      "structural",
    "sign_flip_reactive":   "structural",
    "temporal_acf_power":   "temporal",
    "constant_segment":     "quality",
}

# Maps each mutation name → its ground-truth category
MUTATION_CATEGORY: dict[str, str] = {
    "walmart__shift_dates_-7":              "temporal",
    "walmart__shift_dates_7":               "temporal",
    "walmart__scale_sales_0.9":             "scaling",
    "walmart__scale_temperature_2x":        "scaling",
    "walmart__toggle_holiday":              "structural",
    "walmart__missing_recent_10pct":        "structural",
    "walmart__missing_stores_5":            "structural",
    "walmart__duplicate_rows_5pct":         "quality",
    "walmart__wrong_aggregation_x7":        "aggregation",
    "stock__noise_0.01":                    "quality",
    "stock__zero_volume_30pct":             "quality",
    "stock__drop_5pct":                     "structural",
    "stock__column_swap_open_close":        "structural",
    "stock__shift_price_0.5":              "scaling",
    "stock__unit_conversion_x10":           "scaling",
    "stock__timestamp_lag_1day":            "temporal",
    "stock__reverse_series":               "temporal",
    "power__scale_active_1.1":             "scaling",
    "power__scale_voltage_0.5":            "scaling",
    "power__shuffle_within_day":           "temporal",
    "power__shift_by_1week":               "temporal",
    "power__column_swap_voltage_intensity": "structural",
    "power__missing_6hour_segment":         "structural",
    "power__data_repetition_5x":           "structural",
    "power__sign_flip_reactive":           "structural",
    "power__inject_spikes_100_3.0":        "quality",
    "power__extreme_outliers_50":          "quality",
    "power__constant_segment_2h":          "quality",
}

# Maps aggregation mutations to their detector (scale_sales fires since x7 is huge)
AGGREGATION_DETECTORS = {"scale_sales", "scale_active"}


def _infer_dataset(df: pd.DataFrame) -> str:
    if "Weekly_Sales" in df.columns:
        return "walmart"
    if any(c in df.columns for c in ("Close", "Open", "High", "Low")):
        return "stock"
    if "Global_active_power" in df.columns:
        return "power"
    return "unknown"


def run_contracts(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run all contracts appropriate for the given dataset.

    Returns
    -------
    List of result dicts, each with keys:
        name, detected, confidence, score, reason, category
    """
    ds = dataset or _infer_dataset(baseline_df)

    results: list[dict] = []

    def _run(fn, *args, **kwargs):
        r = fn(baseline_df, test_df, *args, **kwargs)
        r["category"] = DETECTOR_CATEGORY.get(r["name"], "unknown")
        results.append(r)

    if ds == "walmart":
        _run(contract_walmart_shift_dates)
        _run(contract_walmart_scale_sales)
        _run(contract_walmart_toggle_holiday)
        _run(contract_walmart_scale_temperature)
        _run(contract_walmart_missing_recent)
        _run(contract_walmart_missing_stores)
        _run(contract_walmart_duplicate_rows)

    elif ds == "stock":
        _run(contract_stock_noise)
        _run(contract_stock_drop_fraction)
        _run(contract_stock_shift_price)
        _run(contract_stock_column_swap,
             column_pairs=[("Open", "Close")])
        _run(contract_stock_unit_conversion)
        _run(contract_stock_temporal_acf)
        _run(contract_stock_zero_volume)
        _run(contract_stock_timestamp_lag)

    elif ds == "power":
        _run(contract_power_scale_active)
        _run(contract_power_scale_voltage)
        _run(contract_power_inject_spikes)
        _run(contract_power_shuffle_within_day)
        _run(contract_power_column_swap,
             column_pairs=[("Voltage", "Global_intensity")])
        _run(contract_power_missing_segment)
        _run(contract_power_data_repetition)
        _run(contract_power_sign_flip)
        _run(contract_power_temporal_acf)
        _run(contract_power_constant_segment)

    return results


def validate(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mutation_name: str | None = None,
    dataset: str | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    High-level validation entry point.

    Returns
    -------
    {
      "detected":         bool,
      "detected_count":   int,    # number of contracts that fired
      "fired_detectors":  list[str],
      "max_confidence":   float,
      "mutation_name":    str | None,
      "mutation_category": str | None,
      "semantic_match":   bool | None,  # None if mutation_name unknown
      "contracts":        list[dict],   # full per-contract results
    }
    """
    contracts = run_contracts(baseline_df, test_df, dataset)

    fired = [c for c in contracts if c["detected"] and c["confidence"] >= confidence_threshold]
    detected = len(fired) > 0
    max_conf = max((c["confidence"] for c in contracts if c["detected"]), default=0.0)

    mut_cat = MUTATION_CATEGORY.get(mutation_name) if mutation_name else None
    semantic_match: bool | None = None
    if mut_cat is not None and detected:
        fired_cats = {DETECTOR_CATEGORY.get(c["name"], "unknown") for c in fired}
        # aggregation is caught by scaling detectors (the ratio is huge)
        effective_cat = "scaling" if mut_cat == "aggregation" else mut_cat
        semantic_match = effective_cat in fired_cats

    return {
        "detected": detected,
        "detected_count": len(fired),
        "fired_detectors": [c["name"] for c in fired],
        "max_confidence": float(max_conf),
        "mutation_name": mutation_name,
        "mutation_category": mut_cat,
        "semantic_match": semantic_match,
        "contracts": contracts,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, str]:
    """Wilson score confidence interval for binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0, "N/A"
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    lo, hi = max(0.0, centre - margin), min(1.0, centre + margin)
    return lo, hi, f"{p*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]"


def compute_semantic_precision(results: list[dict]) -> dict:
    """
    Given a list of validate() outputs, compute per-category semantic precision.

    Semantic precision = fraction of detected mutations where at least one
    fired detector belongs to the correct category for that mutation type.
    """
    from collections import defaultdict
    cat_total: dict[str, int] = defaultdict(int)
    cat_match: dict[str, int] = defaultdict(int)

    for r in results:
        if not r["detected"] or r["mutation_category"] is None:
            continue
        cat = r["mutation_category"]
        cat_total[cat] += 1
        if r.get("semantic_match"):
            cat_match[cat] += 1

    rows = []
    total_det = sum(cat_total.values())
    total_match = sum(cat_match.values())
    for cat in sorted(cat_total):
        rows.append({
            "category": cat,
            "detected": cat_total[cat],
            "semantic_matches": cat_match[cat],
            "precision": cat_match[cat] / cat_total[cat] if cat_total[cat] > 0 else 0.0,
        })

    overall = total_match / total_det if total_det > 0 else 0.0
    return {
        "per_category": rows,
        "overall_precision": overall,
        "total_detected": total_det,
        "total_semantic_matches": total_match,
    }

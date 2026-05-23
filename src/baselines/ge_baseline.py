"""
src/baselines/ge_baseline.py
==============================
Great Expectations Comprehensive Baseline.

This baseline mimics what a practitioner would configure using the
Great Expectations (GE) framework:  a suite of purely statistical
"expectations" calibrated once on clean reference data, then evaluated
against every incoming (possibly mutated) dataset.

Design philosophy
-----------------
GE is schema-first and distribution-aware, but it has *no* semantic
reasoning about what the columns mean. It checks numerical invariants
(row counts, means, stds, ranges, null rates, value sets) derived from
a one-time profiling pass over the baseline.  Any expectation that fails
votes for "mutation detected."

We deliberately mirror the generous tolerances a practitioner would use
in production to avoid alert fatigue:
  - Row count: ±15 %
  - Column mean: ±20 % (±30 % for stock prices due to natural drift)
  - Column std: within [50 %, 200 %] of baseline std
  - Zero fraction: increase by ≤ 20 pp before flagging

A mutation is detected if ANY expectation fails (OR logic), consistent
with how GE suites are typically configured in production.

References
----------
Shankar et al. (2022). "Operationalizing Machine Learning: An Interview
Study." arXiv:2209.09125  — discusses GE-style monitoring in practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return names of all numeric columns (excluding datetime)."""
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]


def _profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    One-time profiling pass over a clean baseline DataFrame.

    Returns a dict of per-column statistical summaries used to calibrate
    the expectation suite.
    """
    profile: dict[str, Any] = {
        "row_count": len(df),
        "columns": {},
    }
    for col in _numeric_cols(df):
        arr = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(arr) == 0:
            continue
        profile["columns"][col] = {
            "mean":       float(np.mean(arr)),
            "std":        float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median":     float(np.median(arr)),
            "min":        float(np.min(arr)),
            "max":        float(np.max(arr)),
            "null_rate":  float(df[col].isna().mean()),
            "zero_frac":  float((arr == 0).mean()),
            "q01":        float(np.percentile(arr, 1)),
            "q99":        float(np.percentile(arr, 99)),
            "n_unique":   int(pd.Series(arr).nunique()),
        }

    # Categorical / ID columns (store IDs, etc.)
    profile["categorical"] = {}
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("int"):
            nuniq = df[col].nunique()
            if 1 < nuniq <= 200:      # treat as categorical if small enough cardinality
                profile["categorical"][col] = set(df[col].dropna().unique().tolist())

    # Date range
    for dc in ("DateTime", "Date", "date", "datetime"):
        if dc in df.columns:
            dates = pd.to_datetime(df[dc], errors="coerce").dropna()
            if len(dates) > 0:
                profile["date_col"]  = dc
                profile["date_min"]  = dates.min()
                profile["date_max"]  = dates.max()
                profile["date_span"] = (dates.max() - dates.min()).total_seconds() / 86400.0
            break

    return profile


# ---------------------------------------------------------------------------
# Expectation checks
# ---------------------------------------------------------------------------

def _check_row_count(
    profile: dict,
    test_df: pd.DataFrame,
    tolerance: float = 0.15,
) -> list[dict]:
    """Row count within ±tolerance of baseline row count."""
    failures = []
    expected = profile["row_count"]
    actual   = len(test_df)
    if expected == 0:
        return failures
    ratio = actual / expected
    if not (1 - tolerance <= ratio <= 1 + tolerance):
        failures.append({
            "expectation": "expect_table_row_count_to_be_between",
            "column":      "__table__",
            "expected":    f"{expected * (1 - tolerance):.0f}–{expected * (1 + tolerance):.0f}",
            "observed":    actual,
            "score":       abs(ratio - 1.0),
        })
    return failures


def _check_column_mean(
    profile: dict,
    test_df: pd.DataFrame,
    tolerance: float = 0.20,
    stock_price_cols: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close"),
) -> list[dict]:
    """Column mean within ±tolerance of baseline mean."""
    failures = []
    for col, stats in profile["columns"].items():
        if col not in test_df.columns:
            continue
        base_mean = stats["mean"]
        if abs(base_mean) < 1e-8:
            continue
        arr = pd.to_numeric(test_df[col], errors="coerce").dropna().values
        if len(arr) == 0:
            continue
        test_mean = float(np.mean(arr))
        rel_diff  = abs(test_mean - base_mean) / abs(base_mean)
        # Stock price columns are expected to drift; use a generous tolerance
        eff_tol = 0.30 if col in stock_price_cols else tolerance
        if rel_diff > eff_tol:
            failures.append({
                "expectation": "expect_column_mean_to_be_between",
                "column":      col,
                "expected":    f"{base_mean * (1 - eff_tol):.4f}–{base_mean * (1 + eff_tol):.4f}",
                "observed":    test_mean,
                "score":       rel_diff,
            })
    return failures


def _check_column_std(
    profile: dict,
    test_df: pd.DataFrame,
    lo_factor: float = 0.50,
    hi_factor: float = 2.00,
) -> list[dict]:
    """Column std within [lo_factor, hi_factor] × baseline std."""
    failures = []
    for col, stats in profile["columns"].items():
        if col not in test_df.columns:
            continue
        base_std = stats["std"]
        if base_std < 1e-8:
            continue
        arr = pd.to_numeric(test_df[col], errors="coerce").dropna().values
        if len(arr) < 2:
            continue
        test_std = float(np.std(arr, ddof=1))
        ratio    = test_std / base_std
        if not (lo_factor <= ratio <= hi_factor):
            failures.append({
                "expectation": "expect_column_stdev_to_be_between",
                "column":      col,
                "expected":    f"{base_std * lo_factor:.4f}–{base_std * hi_factor:.4f}",
                "observed":    test_std,
                "score":       abs(ratio - 1.0),
            })
    return failures


def _check_null_rate(
    profile: dict,
    test_df: pd.DataFrame,
    max_increase: float = 0.10,
) -> list[dict]:
    """Null rate must not increase by more than max_increase."""
    failures = []
    for col, stats in profile["columns"].items():
        if col not in test_df.columns:
            continue
        base_null = stats["null_rate"]
        test_null = float(test_df[col].isna().mean())
        increase  = test_null - base_null
        if increase > max_increase:
            failures.append({
                "expectation": "expect_column_values_to_not_be_null",
                "column":      col,
                "expected":    f"null_rate ≤ {base_null + max_increase:.3f}",
                "observed":    test_null,
                "score":       increase,
            })
    return failures


def _check_value_range(
    profile: dict,
    test_df: pd.DataFrame,
    pct_threshold: float = 0.99,
) -> list[dict]:
    """
    99 % of test values must fall within [q01_base, q99_base] × expansion factor.
    Deliberately generous to handle small legitimate range expansions.
    """
    failures = []
    expansion = 0.50   # allow 50 % expansion beyond training [q01, q99]
    for col, stats in profile["columns"].items():
        if col not in test_df.columns:
            continue
        lo = stats["q01"] - abs(stats["q01"]) * expansion - 1e-8
        hi = stats["q99"] + abs(stats["q99"]) * expansion + 1e-8
        arr = pd.to_numeric(test_df[col], errors="coerce").dropna().values
        if len(arr) == 0:
            continue
        in_range = float(np.mean((arr >= lo) & (arr <= hi)))
        if in_range < pct_threshold:
            failures.append({
                "expectation": "expect_column_values_to_be_between",
                "column":      col,
                "expected":    f"≥{pct_threshold:.0%} in [{lo:.2f}, {hi:.2f}]",
                "observed":    f"{in_range:.3%}",
                "score":       pct_threshold - in_range,
            })
    return failures


def _check_zero_fraction(
    profile: dict,
    test_df: pd.DataFrame,
    max_increase: float = 0.20,
) -> list[dict]:
    """Zero fraction increase > max_increase ⟹ sparse-data corruption."""
    failures = []
    for col, stats in profile["columns"].items():
        if col not in test_df.columns:
            continue
        base_zero = stats["zero_frac"]
        arr = pd.to_numeric(test_df[col], errors="coerce").fillna(0).values
        test_zero = float((arr == 0).mean()) if len(arr) > 0 else 0.0
        increase  = test_zero - base_zero
        if increase > max_increase:
            failures.append({
                "expectation": "expect_column_values_to_not_be_zero",
                "column":      col,
                "expected":    f"zero_frac ≤ {base_zero + max_increase:.3f}",
                "observed":    test_zero,
                "score":       increase,
            })
    return failures


def _check_categorical_values(
    profile: dict,
    test_df: pd.DataFrame,
    max_missing_frac: float = 0.10,
) -> list[dict]:
    """
    Categorical columns (e.g. Store IDs) should not lose >10 % of their
    unique values relative to baseline.
    """
    failures = []
    for col, base_vals in profile.get("categorical", {}).items():
        if col not in test_df.columns:
            continue
        test_vals = set(test_df[col].dropna().unique().tolist())
        if len(base_vals) == 0:
            continue
        missing_frac = len(base_vals - test_vals) / len(base_vals)
        if missing_frac > max_missing_frac:
            failures.append({
                "expectation": "expect_column_values_to_be_in_set",
                "column":      col,
                "expected":    f"≤{max_missing_frac:.0%} unique values missing",
                "observed":    f"{missing_frac:.1%} missing ({len(base_vals - test_vals)} values)",
                "score":       missing_frac,
            })
    return failures


def _check_sign_distribution(
    profile: dict,
    test_df: pd.DataFrame,
    min_positive_ratio: float = 0.80,
    relevant_cols: tuple[str, ...] = ("Global_reactive_power",),
) -> list[dict]:
    """
    For columns that should be predominantly positive (e.g. reactive power),
    a sign flip will invert the positive-fraction.  This is a targeted GE
    expectation for the power dataset.
    """
    failures = []
    for col in relevant_cols:
        if col not in test_df.columns or col not in profile["columns"]:
            continue
        arr = pd.to_numeric(test_df[col], errors="coerce").dropna().values
        if len(arr) == 0:
            continue
        base_arr_profile = profile["columns"][col]
        # Only apply if baseline is predominantly positive
        base_positive_frac = 1.0 - float(
            # approximate: if mean >> 0 and median >> 0 we assume mostly positive
            (base_arr_profile["median"] < 0 or base_arr_profile["mean"] < 0)
        )
        if base_positive_frac < min_positive_ratio:
            continue
        test_positive_frac = float((arr > 0).mean())
        if test_positive_frac < 1 - min_positive_ratio:
            failures.append({
                "expectation": "expect_column_values_to_be_positive",
                "column":      col,
                "expected":    f"≥{min_positive_ratio:.0%} positive",
                "observed":    f"{test_positive_frac:.1%} positive",
                "score":       min_positive_ratio - test_positive_frac,
            })
    return failures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ge_baseline(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the full GE-style expectation suite.

    Parameters
    ----------
    baseline_df : clean reference DataFrame (used to calibrate expectations)
    test_df     : incoming DataFrame to validate
    dataset     : optional hint ("walmart" | "stock" | "power") for
                  dataset-specific tolerances
    verbose     : if True, print failed expectations

    Returns
    -------
    {
      "detected":      bool,    # True if any expectation failed
      "n_failed":      int,     # number of failed expectations
      "n_checked":     int,     # total number of expectations checked
      "failures":      list,    # list of failure detail dicts
      "score":         float,   # fraction of expectations that failed
    }
    """
    # Calibrate tolerances by dataset
    mean_tol = 0.20
    if dataset == "stock":
        mean_tol = 0.30   # stock prices drift legitimately over time

    profile = _profile(baseline_df)

    all_failures: list[dict] = []
    all_failures += _check_row_count(profile, test_df, tolerance=0.15)
    all_failures += _check_column_mean(profile, test_df, tolerance=mean_tol)
    all_failures += _check_column_std(profile, test_df, lo_factor=0.50, hi_factor=2.00)
    all_failures += _check_null_rate(profile, test_df, max_increase=0.10)
    all_failures += _check_value_range(profile, test_df, pct_threshold=0.99)
    all_failures += _check_zero_fraction(profile, test_df, max_increase=0.20)
    all_failures += _check_categorical_values(profile, test_df, max_missing_frac=0.10)
    all_failures += _check_sign_distribution(profile, test_df)

    # Rough total expectation count: 1 row-count + per-col checks × n_cols
    n_cols  = len(profile["columns"])
    n_checked = 1 + n_cols * 6 + len(profile.get("categorical", {})) + 1

    detected = len(all_failures) > 0
    score    = len(all_failures) / max(n_checked, 1)

    if verbose and all_failures:
        print(f"[GE] {len(all_failures)} expectation(s) failed:")
        for f in all_failures:
            print(f"  [{f['expectation']}] col={f['column']}  "
                  f"expected={f['expected']}  observed={f['observed']}")

    return {
        "detected":  bool(detected),
        "n_failed":  len(all_failures),
        "n_checked": n_checked,
        "score":     float(score),
        "failures":  all_failures,
    }

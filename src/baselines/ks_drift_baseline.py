"""
src/baselines/ks_drift_baseline.py
====================================
Original KS-test-only baseline.  Kept verbatim from v1 for comparison.
A mutation is detected if ANY numeric column shows significant KS drift
at p < 0.01 (Bonferroni-corrected across columns).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def run_ks_baseline(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 0.01,
) -> dict:
    cols = [c for c in baseline_df.columns if pd.api.types.is_numeric_dtype(baseline_df[c])]
    p_values: dict[str, float] = {}
    for c in cols:
        a = pd.to_numeric(baseline_df[c], errors="coerce").dropna().values
        b = pd.to_numeric(test_df[c], errors="coerce").dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        _, p = ks_2samp(a, b)
        p_values[c] = float(p)

    if not p_values:
        return {"detected": False, "score": None, "p_values": {}}

    # Bonferroni correction
    min_p = min(p_values.values())
    detected = min_p < alpha / len(p_values)
    return {"detected": bool(detected), "score": min_p, "p_values": p_values}

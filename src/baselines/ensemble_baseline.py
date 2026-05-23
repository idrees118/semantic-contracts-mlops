"""
src/baselines/ensemble_baseline.py
====================================
Ensemble Statistical Baseline — the new, stronger baseline for v2.

Combines five independent statistical tests into a single detector via
majority-vote (any-one) aggregation:

  1. KS test          — Kolmogorov-Smirnov on each numeric column
  2. Jensen-Shannon   — JSD on binned empirical distributions
  3. PSI              — Population Stability Index (binned)
  4. Corr-matrix      — Frobenius norm of Pearson correlation matrix change
  5. ACF comparison   — Mean absolute difference of autocorrelation functions

A mutation is "detected" if ANY ONE of the five tests fires.

This is deliberately more sensitive than the original KS-only baseline, but
still purely statistical — it has no access to domain-specific semantic rules.
It will detect distribution changes, correlation changes, and temporal-order
changes, but cannot reason about meaning (e.g., what a holiday flag means, or
that a rolling 7-day average should be consistent with the raw sales).

References
----------
- Population Stability Index: Yurdakul (2018), 
  "Statistical Properties of Population Stability Index"
- Jensen-Shannon divergence: Lin (1991), IEEE Trans. IT
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.special import rel_entr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _safe_hist(arr: np.ndarray, bins: int = 10) -> np.ndarray:
    """Return normalised histogram with Laplace smoothing to avoid log(0)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.ones(bins) / bins
    h, _ = np.histogram(arr, bins=bins)
    h = h.astype(float) + 1e-8     # Laplace smooth
    return h / h.sum()


def _acf(arr: np.ndarray, nlags: int = 20) -> np.ndarray:
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < nlags + 2:
        return np.zeros(nlags)
    arr = arr - arr.mean()
    var = float(np.var(arr))
    if var == 0:
        return np.zeros(nlags)
    return np.array([float(np.mean(arr[:-k] * arr[k:])) / var for k in range(1, nlags + 1)])


def _dt_col(df: pd.DataFrame) -> str | None:
    for c in ("DateTime", "Date", "date", "datetime"):
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Test 1: KS test (column-wise)
# ---------------------------------------------------------------------------

def test_ks(base: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.01) -> dict:
    """
    Two-sample KS test on each numeric column.
    Detected if ANY column shows significant drift.
    """
    cols = _numeric_cols(base)
    p_values = {}
    for c in cols:
        a = pd.to_numeric(base[c], errors="coerce").dropna().values
        b = pd.to_numeric(test[c], errors="coerce").dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        _, p = ks_2samp(a, b)
        p_values[c] = float(p)

    if not p_values:
        return {"test": "ks", "detected": False, "score": None}

    min_p = min(p_values.values())
    # Bonferroni correction: require at least one column's p < alpha / n_cols
    n = len(p_values)
    detected = min_p < (alpha / n)
    return {"test": "ks", "detected": detected, "score": min_p, "p_values": p_values}


# ---------------------------------------------------------------------------
# Test 2: Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def test_jsd(base: pd.DataFrame, test: pd.DataFrame,
             threshold: float = 0.05, bins: int = 20) -> dict:
    """
    Jensen-Shannon divergence on binned column distributions.
    JSD ∈ [0, log(2)]; values > threshold flag a mutation.
    We use common bins from the combined data to ensure comparability.
    """
    cols = _numeric_cols(base)
    jsds = {}
    for c in cols:
        a = pd.to_numeric(base[c], errors="coerce").dropna().values
        b = pd.to_numeric(test[c], errors="coerce").dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        combined = np.concatenate([a, b])
        edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        p = np.histogram(a, bins=edges)[0].astype(float) + 1e-8
        q = np.histogram(b, bins=edges)[0].astype(float) + 1e-8
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = float(0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum())
        jsds[c] = jsd

    if not jsds:
        return {"test": "jsd", "detected": False, "score": None}

    max_jsd = max(jsds.values())
    detected = max_jsd > threshold
    return {"test": "jsd", "detected": detected, "score": max_jsd, "per_col": jsds}


# ---------------------------------------------------------------------------
# Test 3: Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def _psi_col(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI for a single column using quantile-based bins from expected."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    quantiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return 0.0

    def bucket_fracs(arr):
        counts = np.histogram(arr, bins=quantiles)[0].astype(float)
        counts += 1e-8
        return counts / counts.sum()

    p = bucket_fracs(expected)
    q = bucket_fracs(actual)
    psi = float(np.sum((q - p) * np.log(q / p)))
    return psi


def test_psi(base: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.10) -> dict:
    """
    PSI > 0.10 is typically considered "moderate shift";
    PSI > 0.25 is "significant shift".
    We use 0.10 as the threshold (conservative).
    """
    cols = _numeric_cols(base)
    psis = {}
    for c in cols:
        a = pd.to_numeric(base[c], errors="coerce").dropna().values
        b = pd.to_numeric(test[c], errors="coerce").dropna().values
        psis[c] = _psi_col(a, b)

    if not psis:
        return {"test": "psi", "detected": False, "score": None}

    max_psi = max(psis.values())
    detected = max_psi > threshold
    return {"test": "psi", "detected": detected, "score": max_psi, "per_col": psis}


# ---------------------------------------------------------------------------
# Test 4: Correlation matrix Frobenius distance
# ---------------------------------------------------------------------------

def test_corr_matrix(
    base: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.20,     # Frobenius norm of the diff / sqrt(n_cols^2)
    min_cols: int = 3,
) -> dict:
    """
    Compare Pearson inter-column correlation matrices.
    A large Frobenius distance indicates structural relationship changes
    (e.g., column swaps change the entire correlation structure).

    Normalised Frobenius = ||C_base - C_test||_F / n_cols
    """
    cols = [c for c in _numeric_cols(base) if c in test.columns]
    if len(cols) < min_cols:
        return {"test": "corr_matrix", "detected": False, "score": None}

    a = base[cols].apply(pd.to_numeric, errors="coerce").dropna()
    b = test[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10:
        return {"test": "corr_matrix", "detected": False, "score": None}

    # Use the minimum-length shared prefix to ensure equal sample count
    n = min(len(a), len(b))
    try:
        ca = np.corrcoef(a.values[:n].T)
        cb = np.corrcoef(b.values[:n].T)
    except Exception:
        return {"test": "corr_matrix", "detected": False, "score": None}

    diff = ca - cb
    frob = float(np.linalg.norm(diff, ord="fro")) / len(cols)
    detected = frob > threshold
    return {"test": "corr_matrix", "detected": detected, "score": frob}


# ---------------------------------------------------------------------------
# Test 5: ACF comparison
# ---------------------------------------------------------------------------

def test_acf_comparison(
    base: pd.DataFrame,
    test: pd.DataFrame,
    nlags: int = 20,
    threshold: float = 0.08,
) -> dict:
    """
    Compare autocorrelation functions across numeric columns.
    Mean absolute difference > threshold ⟹ temporal mutation detected.

    This catches column shuffles, series reversals, and timestamp lags
    that leave marginal distributions unchanged.
    """
    cols = _numeric_cols(base)
    acf_diffs = {}
    for c in cols:
        a = pd.to_numeric(base[c], errors="coerce").ffill().dropna().values
        b = pd.to_numeric(test[c], errors="coerce").ffill().dropna().values
        if min(len(a), len(b)) < nlags + 10:
            continue
        acf_a = _acf(a, nlags)
        acf_b = _acf(b, nlags)
        acf_diffs[c] = float(np.mean(np.abs(acf_a - acf_b)))

    if not acf_diffs:
        return {"test": "acf", "detected": False, "score": None}

    max_diff = max(acf_diffs.values())
    detected = max_diff > threshold
    return {"test": "acf", "detected": detected, "score": max_diff, "per_col": acf_diffs}


# ---------------------------------------------------------------------------
# Top-level ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble_baseline(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    voting: str = "any",    # "any" = OR across all tests | "majority" = ≥3/5
    verbose: bool = False,
) -> dict:
    """
    Run all five statistical tests and combine with the specified voting rule.

    Parameters
    ----------
    voting : "any" fires if ANY one test detects a mutation (sensitive, ~false
             positive aware but maximises recall).
             "majority" fires only if ≥3/5 tests agree (conservative).

    Returns
    -------
    {
      "detected":  bool,
      "score":     float (fraction of tests that fired),
      "tests":     {test_name: result_dict, ...},
      "voting":    str
    }
    """
    tests = {
        "ks":          test_ks(baseline_df, test_df),
        "jsd":         test_jsd(baseline_df, test_df),
        "psi":         test_psi(baseline_df, test_df),
        "corr_matrix": test_corr_matrix(baseline_df, test_df),
        "acf":         test_acf_comparison(baseline_df, test_df),
    }

    fired = [name for name, r in tests.items() if r.get("detected", False)]
    n_fired = len(fired)

    if voting == "any":
        detected = n_fired >= 1
    elif voting == "majority":
        detected = n_fired >= 3
    else:
        raise ValueError(f"Unknown voting rule: {voting!r}")

    score = n_fired / len(tests)

    if verbose:
        print(f"Ensemble ({voting}): {n_fired}/5 tests fired — detected={detected}")
        for name, r in tests.items():
            flag = "✓" if r.get("detected") else "✗"
            print(f"  [{flag}] {name}: score={r.get('score')}")

    return {
        "detected": bool(detected),
        "score": float(score),
        "n_fired": n_fired,
        "fired_tests": fired,
        "tests": tests,
        "voting": voting,
    }

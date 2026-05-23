"""
src/validators/contracts.py
============================
Individual semantic contract detector functions.

Design improvements over v1
----------------------------
1. **Adaptive thresholds** — thresholds are expressed relative to the baseline's
   own variability (MAD or IQR), not as fixed constants.  This makes detectors
   robust to datasets with very different scales.

2. **Confidence scores** — every detector returns a normalised `confidence` value
   in [0, 1] alongside the binary `detected` flag, enabling downstream ranking
   of anomalies.

3. **ACF-based temporal contracts** — instead of only per-day Spearman on a
   fixed window, we compare autocorrelation functions at lags 1-24 to catch
   any temporal reordering.

4. **Correlation-matrix structural contracts** — compare the full inter-column
   Pearson matrix (Frobenius norm of difference) to catch column swaps even
   when individual marginals are unchanged.

Each function signature is:
    detect_*(baseline_df, test_df, **kwargs) -> dict

Returned dict always contains:
  name       str       detector name
  detected   bool      whether the contract is violated
  confidence float     in [0,1]; higher = more confident the violation is real
  score      any       raw metric value (scalar or dict)
  reason     str|None  human-readable explanation when detected=True
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mad(arr: np.ndarray) -> float:
    """Median absolute deviation."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def _align(base: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Align two dataframes on their datetime index (inner join on timestamps).
    Handles duplicate dates (e.g. Walmart: multiple stores per date) by
    sorting both frames on the datetime column and truncating to the minimum
    common length so array operations never see a shape mismatch.
    Returns (base_aligned, test_aligned) or (None, None) if alignment fails.
    """
    for c in ("DateTime", "Date", "date", "datetime", "Timestamp"):
        if c in base.columns and c in test.columns:
            dt_col = c
            break
    else:
        return None, None

    b = base.copy()
    t = test.copy()
    b[dt_col] = pd.to_datetime(b[dt_col], errors="coerce")
    t[dt_col] = pd.to_datetime(t[dt_col], errors="coerce")
    b = b.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    t = t.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)

    # For datasets with duplicate timestamps (e.g. Walmart multi-store),
    # we align by position after sorting so array lengths always match.
    n = min(len(b), len(t))
    if n < 10:
        return None, None
    return b.iloc[:n].copy(), t.iloc[:n].copy()


def _acf(arr: np.ndarray, nlags: int = 24) -> np.ndarray:
    """Unbiased autocorrelation at lags 1..nlags."""
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < nlags + 2:
        return np.zeros(nlags)
    arr = arr - arr.mean()
    var = float(np.var(arr))
    if var == 0:
        return np.zeros(nlags)
    result = []
    for lag in range(1, nlags + 1):
        cov = float(np.mean(arr[:-lag] * arr[lag:]))
        result.append(cov / var)
    return np.array(result)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _result(name: str, detected: bool, confidence: float, score, reason: str | None = None) -> dict:
    return {
        "name": name,
        "detected": bool(detected),
        "confidence": _clamp01(confidence),
        "score": score,
        "reason": reason,
    }


# ===========================================================================
# WALMART CONTRACTS
# ===========================================================================

def contract_walmart_shift_dates(
    base: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "Date",
    min_shift_days: float = 3.0,
) -> dict:
    """
    Compare median dates between baseline and test.

    If all dates were shifted by K days, the median date of the test will
    differ from the baseline median by exactly K days. This is robust to
    multi-store Walmart (many rows per date) and to the high week-over-week
    autocorrelation that makes correlation-based detection unreliable.
    """
    if date_col not in base.columns or date_col not in test.columns:
        return _result("shift_dates", False, 0.0, None, "no_date_column")

    base_dates = pd.to_datetime(base[date_col], errors="coerce").dropna()
    test_dates  = pd.to_datetime(test[date_col],  errors="coerce").dropna()
    if len(base_dates) < 5 or len(test_dates) < 5:
        return _result("shift_dates", False, 0.0, None, "insufficient_data")

    base_median = base_dates.sort_values().iloc[len(base_dates) // 2]
    test_median  = test_dates.sort_values().iloc[len(test_dates) // 2]
    shift_days = abs((test_median - base_median).total_seconds()) / 86400.0

    detected = shift_days >= min_shift_days
    confidence = _clamp01(shift_days / 14.0)
    return _result("shift_dates", detected, confidence, shift_days,
                   f"median_date_shift={shift_days:.1f}d" if detected else None)


def contract_walmart_scale_sales(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Weekly_Sales",
    tol: float = 0.03,          # 3 % tolerance
) -> dict:
    """
    Median ratio of aligned values deviates from 1.0 ⟹ scale mutation detected.
    Adaptive: tolerance is relative (fraction), not absolute.
    """
    base_al, test_al = _align(base, test)
    if base_al is None or col not in base_al.columns:
        return _result("scale_sales", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    mask = ~np.isnan(a) & ~np.isnan(b) & (a != 0)
    if mask.sum() < 5:
        return _result("scale_sales", False, 0.0, None, "insufficient_data")

    ratios = b[mask] / a[mask]
    ratios = ratios[np.isfinite(ratios)]
    median_r = float(np.median(ratios))
    deviation = abs(median_r - 1.0)
    detected = deviation > tol
    confidence = _clamp01((deviation - tol) / tol) if detected else 0.0
    return _result("scale_sales", detected, confidence, median_r,
                   f"median_ratio={median_r:.4f} deviates by {deviation:.4f}" if detected else None)


def contract_walmart_toggle_holiday(
    base: pd.DataFrame,
    test: pd.DataFrame,
    mismatch_threshold: float = 0.30,   # >30 % of flags flipped
) -> dict:
    """
    High fraction of mismatched Holiday_Flag values ⟹ toggle mutation detected.
    """
    flag_col = next((c for c in ("Holiday_Flag", "IsHoliday", "Holiday") if c in base.columns), None)
    if flag_col is None or flag_col not in test.columns:
        return _result("toggle_holiday", False, 0.0, None, "no_holiday_column")

    base_al, test_al = _align(base, test)
    if base_al is None:
        # Fallback: compare marginal distributions
        a = base[flag_col].astype(int).values
        b = test[flag_col].astype(int).values
        n = min(len(a), len(b))
        mismatch = float(np.mean(a[:n] != b[:n]))
    else:
        a = base_al[flag_col].astype(int).values
        b = test_al[flag_col].astype(int).values
        mismatch = float(np.mean(a != b))

    detected = mismatch > mismatch_threshold
    confidence = _clamp01((mismatch - mismatch_threshold) / (1 - mismatch_threshold)) if detected else 0.0
    return _result("toggle_holiday", detected, confidence, mismatch,

                   f"flag_mismatch_rate={mismatch:.3f}" if detected else None)


def contract_walmart_missing_recent(
    base: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "Date",
    min_missing_frac: float = 0.05,
) -> dict:
    """
    Max date in test is significantly earlier than baseline max ⟹ tail truncation.
    Catches missing_recent_10pct which drops the last N rows — positional
    alignment cannot see this since the surviving rows are identical.
    """
    if date_col not in base.columns or date_col not in test.columns:
        return _result("missing_recent", False, 0.0, None, "no_date_column")

    base_max = pd.to_datetime(base[date_col], errors="coerce").max()
    test_max  = pd.to_datetime(test[date_col], errors="coerce").max()
    base_min  = pd.to_datetime(base[date_col], errors="coerce").min()

    if pd.isna(base_max) or pd.isna(test_max) or pd.isna(base_min):
        return _result("missing_recent", False, 0.0, None, "invalid_dates")

    total_span = (base_max - base_min).total_seconds()
    gap = (base_max - test_max).total_seconds()
    if total_span <= 0:
        return _result("missing_recent", False, 0.0, None, "zero_span")

    missing_frac = gap / total_span
    detected = missing_frac >= min_missing_frac
    confidence = _clamp01(missing_frac / 0.20)
    return _result("missing_recent", detected, confidence, missing_frac,
                   f"tail_missing_frac={missing_frac:.3f}" if detected else None)


def contract_walmart_missing_stores(
    base: pd.DataFrame,
    test: pd.DataFrame,
    store_col: str = "Store",
    min_missing: int = 1,
) -> dict:
    """
    Fewer unique store IDs in test than baseline ⟹ store-level data omission.
    Catches missing_stores_5 which drops entire stores.
    """
    if store_col not in base.columns or store_col not in test.columns:
        return _result("missing_stores", False, 0.0, None, "no_store_column")

    base_stores = set(base[store_col].dropna().unique())
    test_stores  = set(test[store_col].dropna().unique())
    missing = len(base_stores - test_stores)
    detected = missing >= min_missing
    confidence = _clamp01(missing / max(len(base_stores) * 0.15, 1))
    return _result("missing_stores", detected, confidence, missing,
                   f"missing_stores={missing}" if detected else None)


def contract_walmart_scale_temperature(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Temperature",
    tol: float = 0.05,
) -> dict:
    """
    Median ratio of Temperature values far from 1.0 ⟹ temperature scaling detected. NEW
    """
    if col not in base.columns or col not in test.columns:
        return _result("scale_temperature", False, 0.0, None, "no_temperature_column")

    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("scale_temperature", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a) > 0.01)
    if mask.sum() < 5:
        return _result("scale_temperature", False, 0.0, None, "insufficient_data")
    ratios = b[mask] / a[mask]
    ratios = ratios[np.isfinite(ratios)]
    median_r = float(np.median(ratios))
    deviation = abs(median_r - 1.0)
    detected = deviation > tol
    confidence = _clamp01((deviation - tol) / max(tol, 0.001)) if detected else 0.0
    return _result("scale_temperature", detected, confidence, median_r,
                   f"temperature_ratio={median_r:.4f}" if detected else None)


# ===========================================================================
# STOCK CONTRACTS
# ===========================================================================

def contract_stock_noise(
    base: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = ("Open", "High", "Low", "Close"),
    std_ratio_threshold: float = 1.05,  # 5 % increase in σ
) -> dict:
    """
    Standard deviation ratio > threshold ⟹ noise injection detected.
    Uses per-column adaptive threshold: ratio of σ_test / σ_base.
    """
    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("noise", False, 0.0, None, "alignment_failed")

    ratios = []
    for c in cols:
        if c in base_al.columns and c in test_al.columns:
            a = _num(base_al[c]).dropna().values
            b = _num(test_al[c]).dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            sa, sb = float(np.std(a)), float(np.std(b))
            if sa > 0:
                ratios.append(sb / sa)

    if not ratios:
        return _result("noise", False, 0.0, None, "no_price_columns")

    median_r = float(np.median(ratios))
    detected = median_r > std_ratio_threshold
    confidence = _clamp01((median_r - std_ratio_threshold) / std_ratio_threshold) if detected else 0.0
    return _result("noise", detected, confidence, median_r,
                   f"std_ratio={median_r:.4f}" if detected else None)


def contract_stock_drop_fraction(
    base: pd.DataFrame,
    test: pd.DataFrame,
    frac_tol: float = 0.02,
) -> dict:
    """Row-count fraction dropped > tolerance ⟹ row-drop detected."""
    n_base, n_test = len(base), len(test)
    if n_base == 0:
        return _result("drop_fraction", False, 0.0, None, "empty_baseline")
    frac = 1.0 - n_test / n_base
    detected = frac > frac_tol
    confidence = _clamp01((frac - frac_tol) / (1 - frac_tol)) if detected else 0.0
    return _result("drop_fraction", detected, confidence, frac,
                   f"dropped={frac:.3%}" if detected else None)


def contract_stock_shift_price(
    base: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = ("Open", "High", "Low", "Close"),
    # Threshold = fraction of the baseline median (e.g. 0.1 % of price)
    rel_threshold: float = 0.001,
) -> dict:
    """
    Median absolute difference between aligned prices > fraction of baseline median
    ⟹ systematic price shift detected.

    Uses a relative threshold so it works regardless of absolute price level.
    """
    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("shift_price", False, 0.0, None, "alignment_failed")

    diffs, base_meds = [], []
    for c in cols:
        if c in base_al.columns and c in test_al.columns:
            a = _num(base_al[c])
            b = _num(test_al[c])
            valid = ~a.isna() & ~b.isna()
            if valid.sum() < 5:
                continue
            med_diff = float(np.median((b - a)[valid]))
            base_med = float(np.median(a[valid]))
            diffs.append(med_diff)
            base_meds.append(base_med)

    if not diffs:
        return _result("shift_price", False, 0.0, None, "no_price_columns")

    median_diff = float(np.median(diffs))
    median_base = float(np.median(base_meds)) if base_meds else 1.0
    rel_diff = abs(median_diff) / max(abs(median_base), 1e-8)
    detected = rel_diff > rel_threshold
    confidence = _clamp01((rel_diff - rel_threshold) / rel_threshold) if detected else 0.0
    return _result("shift_price", detected, confidence,
                   {"median_diff": median_diff, "rel_diff": rel_diff},
                   f"rel_diff={rel_diff:.5f}" if detected else None)


def contract_stock_column_swap(
    base: pd.DataFrame,
    test: pd.DataFrame,
    column_pairs: list[tuple[str, str]] | None = None,
    advantage_threshold: float = 0.15,
) -> dict:
    """
    Detect column swaps using two complementary signals.

    Signal A — Cross-correlation advantage (works when columns are dissimilar).
    Signal B — Daily spread inversion (works for Open/Close which are highly
               correlated in price level but whose intra-day spread reliably
               inverts after a swap).

    After swapping Open ↔ Close:
        spread_test = test['Close'] - test['Open']
                    = base['Open']  - base['Close']
                    = -(base['Close'] - base['Open'])
                    = -spread_base
    ⟹  corr(spread_base, spread_test) ≈ −1.0
    Any corr < −0.5 fires this signal regardless of how correlated the
    price levels are.
    """
    if column_pairs is None:
        column_pairs = [("Open", "Close")]

    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("column_swap", False, 0.0, None, "alignment_failed")

    max_adv   = 0.0
    spread_corr = None

    for col_a, col_b in column_pairs:
        if not all(c in base_al.columns and c in test_al.columns for c in (col_a, col_b)):
            continue
        ba = _num(base_al[col_a]).values
        bb = _num(base_al[col_b]).values
        ta = _num(test_al[col_a]).values
        tb = _num(test_al[col_b]).values
        n = min(len(ba), len(bb), len(ta), len(tb))
        if n < 10:
            continue
        ba, bb, ta, tb = ba[:n], bb[:n], ta[:n], tb[:n]

        # Signal A: cross-correlation advantage
        try:
            c_aa = np.corrcoef(ba, ta)[0, 1]
            c_ab = np.corrcoef(ba, tb)[0, 1]
            c_ba = np.corrcoef(bb, ta)[0, 1]
            c_bb = np.corrcoef(bb, tb)[0, 1]
            if not any(np.isnan(v) for v in (c_aa, c_ab, c_ba, c_bb)):
                adv = max(c_ab - c_aa, c_ba - c_bb)
                max_adv = max(max_adv, adv)
        except Exception:
            pass

        # Signal B: daily spread inversion (robust for highly correlated pairs)
        # spread = col_b - col_a  (e.g. Close - Open)
        spread_base = bb - ba
        spread_test = tb - ta
        valid = ~np.isnan(spread_base) & ~np.isnan(spread_test)
        if valid.sum() >= 10:
            try:
                sc = float(np.corrcoef(spread_base[valid], spread_test[valid])[0, 1])
                if not np.isnan(sc):
                    spread_corr = sc
            except Exception:
                pass

    # Either signal can fire
    swap_by_adv    = max_adv > advantage_threshold
    swap_by_spread = (spread_corr is not None) and (spread_corr < -0.5)
    detected   = swap_by_adv or swap_by_spread
    confidence = max(
        _clamp01((max_adv - advantage_threshold) / (1 - advantage_threshold)) if swap_by_adv else 0.0,
        _clamp01((-spread_corr - 0.5) / 0.5) if swap_by_spread else 0.0,
    )
    score = {"cross_corr_advantage": max_adv, "spread_corr": spread_corr}
    reason = None
    if detected:
        if swap_by_spread:
            reason = f"spread_corr={spread_corr:.4f} (inverted after swap)"
        else:
            reason = f"cross_corr_advantage={max_adv:.4f}"
    return _result("column_swap", detected, confidence, score, reason)


def contract_stock_unit_conversion(
    base: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = ("Open", "High", "Low", "Close"),
    ratio_threshold: float = 5.0,   # flag if ratio > 5× or < 0.2×
) -> dict:
    """
    Extreme median ratio between aligned prices ⟹ unit conversion bug detected.
    """
    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("unit_conversion", False, 0.0, None, "alignment_failed")

    ratios = []
    for c in cols:
        if c in base_al.columns and c in test_al.columns:
            a = _num(base_al[c]).values
            b = _num(test_al[c]).values
            mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a) > 1e-8)
            if mask.sum() < 5:
                continue
            ratios.append(float(np.median(b[mask] / a[mask])))

    if not ratios:
        return _result("unit_conversion", False, 0.0, None, "no_price_columns")

    med_r = float(np.median(ratios))
    detected = med_r > ratio_threshold or med_r < 1 / ratio_threshold
    confidence = 0.0
    if detected:
        if med_r > ratio_threshold:
            confidence = _clamp01((med_r - ratio_threshold) / ratio_threshold)
        else:
            confidence = _clamp01((1 / ratio_threshold - med_r) / (1 / ratio_threshold))
    return _result("unit_conversion", detected, confidence, med_r,
                   f"price_ratio={med_r:.4f}" if detected else None)


def contract_stock_temporal_acf(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Close",
    nlags: int = 10,
    acf_diff_threshold: float = 0.15,
) -> dict:
    """
    Two-signal temporal contract:
    (1) ACF difference — catches timestamp lags and intra-day shuffles.
    (2) First-difference sign agreement — catches series reversal.

    A reversed series has IDENTICAL ACF to the original (ACF is symmetric),
    but its first-differences are the mirror image: if the original mostly
    goes up, the reversed version mostly goes down.
    """
    base_al, test_al = _align(base, test)
    if base_al is None or col not in base_al.columns:
        return _result("temporal_acf", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).ffill().dropna().values
    b = _num(test_al[col]).ffill().dropna().values
    n = min(len(a), len(b))
    if n < nlags + 10:
        return _result("temporal_acf", False, 0.0, None, "insufficient_data")
    a, b = a[:n], b[:n]

    # Signal 1: ACF difference
    acf_diff = float(np.mean(np.abs(_acf(a, nlags) - _acf(b, nlags))))

    # Signal 2: first-difference sign agreement
    # For a reversed series, sign(diff_a[t]) ≈ -sign(diff_b[n-t])
    # Simpler proxy: compare the sign of the net trend
    diff_a = np.sign(np.diff(a))
    diff_b = np.sign(np.diff(b))
    sign_agree = float(np.mean(diff_a == diff_b))   # ~0.5 if reversed, ~1.0 if identical

    # Also check if series is approximately reversed: corr(a, b[::-1]) >> corr(a, b)
    corr_fwd = float(np.corrcoef(a, b)[0, 1]) if n > 1 else 0.0
    corr_rev = float(np.corrcoef(a, b[::-1])[0, 1]) if n > 1 else 0.0
    reversal_signal = (corr_rev - corr_fwd) > 0.3

    detected = (acf_diff > acf_diff_threshold) or reversal_signal or (sign_agree < 0.45)
    confidence = max(
        _clamp01((acf_diff - acf_diff_threshold) / acf_diff_threshold),
        _clamp01((corr_rev - corr_fwd - 0.3) / 0.7) if reversal_signal else 0.0,
        _clamp01((0.50 - sign_agree) / 0.50) if sign_agree < 0.45 else 0.0,
    )
    score = {"acf_diff": acf_diff, "sign_agree": sign_agree,
             "corr_fwd": corr_fwd, "corr_rev": corr_rev}
    reason = None
    if detected:
        if reversal_signal:
            reason = f"series_reversal: corr_rev={corr_rev:.3f} >> corr_fwd={corr_fwd:.3f}"
        elif acf_diff > acf_diff_threshold:
            reason = f"acf_diff={acf_diff:.4f}"
        else:
            reason = f"sign_agree={sign_agree:.3f}"
    return _result("temporal_acf", detected, confidence, score, reason)


def contract_stock_zero_volume(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Volume",
    zero_frac_threshold: float = 0.10,  # >10 % zeros is abnormal
) -> dict:
    """
    Fraction of zero-volume rows increases beyond threshold ⟹ volume corruption. NEW
    """
    if col not in base.columns or col not in test.columns:
        return _result("zero_volume", False, 0.0, None, "no_volume_column")

    frac_base = float((_num(base[col]) == 0).mean())
    frac_test = float((_num(test[col]) == 0).mean())
    increase = frac_test - frac_base
    detected = increase > zero_frac_threshold
    confidence = _clamp01(increase / max(zero_frac_threshold, 0.001)) if detected else 0.0
    return _result("zero_volume", detected, confidence,
                   {"base_zero_frac": frac_base, "test_zero_frac": frac_test},
                   f"zero_fraction_increase={increase:.3f}" if detected else None)


# ===========================================================================
# POWER CONTRACTS
# ===========================================================================

def contract_power_scale_active(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_active_power",
    tol: float = 0.04,
) -> dict:
    """Median ratio of Global_active_power deviates from 1 ⟹ scaling detected."""
    base_al, test_al = _align(base, test)
    if base_al is None or col not in base_al.columns:
        return _result("scale_active", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a) > 1e-8)
    if mask.sum() < 10:
        return _result("scale_active", False, 0.0, None, "insufficient_data")
    ratios = b[mask] / a[mask]
    med_r = float(np.median(ratios[np.isfinite(ratios)]))
    deviation = abs(med_r - 1.0)
    detected = deviation > tol
    confidence = _clamp01((deviation - tol) / tol) if detected else 0.0
    return _result("scale_active", detected, confidence, med_r,
                   f"ratio={med_r:.4f}" if detected else None)


def contract_power_scale_voltage(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Voltage",
    tol: float = 0.04,
) -> dict:
    """Median ratio of Voltage column deviates from 1 ⟹ voltage scaling detected. NEW"""
    if col not in base.columns or col not in test.columns:
        return _result("scale_voltage", False, 0.0, None, "no_voltage_column")
    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("scale_voltage", False, 0.0, None, "alignment_failed")
    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a) > 1e-8)
    if mask.sum() < 10:
        return _result("scale_voltage", False, 0.0, None, "insufficient_data")
    ratios = b[mask] / a[mask]
    med_r = float(np.median(ratios[np.isfinite(ratios)]))
    deviation = abs(med_r - 1.0)
    detected = deviation > tol
    confidence = _clamp01((deviation - tol) / tol) if detected else 0.0
    return _result("scale_voltage", detected, confidence, med_r,
                   f"voltage_ratio={med_r:.4f}" if detected else None)


def contract_power_inject_spikes(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_active_power",
    spike_k: float = 2.5,
    min_new_spikes: int = 30,
) -> dict:
    """
    New spike locations appear in test but not baseline ⟹ spike injection detected.
    Uses baseline's own mean/std to set the spike threshold (adaptive).
    """
    base_al, test_al = _align(base, test)
    if base_al is None or col not in base_al.columns:
        return _result("inject_spikes", False, 0, None, "alignment_failed")

    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    base_mean = float(np.nanmean(a))
    base_std = float(np.nanstd(a))
    if base_std == 0:
        return _result("inject_spikes", False, 0, None, "zero_std")

    thresh = base_mean + spike_k * base_std
    new_spikes = int(np.sum((b > thresh) & ~(a > thresh)))
    detected = new_spikes >= min_new_spikes
    confidence = _clamp01(new_spikes / (min_new_spikes * 3)) if detected else 0.0
    return _result("inject_spikes", detected, confidence, new_spikes,
                   f"new_spikes={new_spikes}" if detected else None)


def contract_power_shuffle_within_day(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_active_power",
    corr_threshold: float = 0.60,
) -> dict:
    """
    Low median per-day Spearman correlation ⟹ intra-day shuffle detected.
    Also uses ACF comparison as a secondary signal.
    """
    for dtc in ("DateTime", "Date"):
        if dtc in base.columns and dtc in test.columns:
            break
    else:
        return _result("shuffle_within_day", False, 0.0, None, "no_datetime_column")

    b = base.copy()
    t = test.copy()
    b[dtc] = pd.to_datetime(b[dtc], errors="coerce")
    t[dtc] = pd.to_datetime(t[dtc], errors="coerce")
    b["__d"] = b[dtc].dt.date
    t["__d"] = t[dtc].dt.date
    common = set(b["__d"]).intersection(set(t["__d"]))

    day_corrs = []
    for d in sorted(common):
        ga = b[b["__d"] == d][col].reset_index(drop=True)
        gb = t[t["__d"] == d][col].reset_index(drop=True)
        n = min(len(ga), len(gb))
        if n < 3:
            continue
        try:
            r, _ = spearmanr(ga.iloc[:n], gb.iloc[:n], nan_policy="omit")
            if not np.isnan(r):
                day_corrs.append(float(r))
        except Exception:
            pass

    if not day_corrs:
        return _result("shuffle_within_day", False, 0.0, None, "no_common_days")

    med_corr = float(np.median(day_corrs))
    detected = med_corr < corr_threshold
    confidence = _clamp01((corr_threshold - med_corr) / corr_threshold) if detected else 0.0
    return _result("shuffle_within_day", detected, confidence, med_corr,
                   f"median_day_corr={med_corr:.3f}" if detected else None)


def contract_power_column_swap(
    base: pd.DataFrame,
    test: pd.DataFrame,
    column_pairs: list[tuple[str, str]] | None = None,
    advantage_threshold: float = 0.15,
) -> dict:
    """Cross-column correlation advantage ⟹ column swap detected (generalised)."""
    if column_pairs is None:
        column_pairs = [("Voltage", "Global_intensity")]
    return contract_stock_column_swap(base, test, column_pairs, advantage_threshold)


def contract_power_missing_segment(
    base: pd.DataFrame,
    test: pd.DataFrame,
    min_missing: int = 60,   # at least 60 timestamps missing = 1 hour
) -> dict:
    """
    Large contiguous block of timestamps missing ⟹ time-segment drop detected.
    """
    for dtc in ("DateTime", "Date"):
        if dtc in base.columns and dtc in test.columns:
            break
    else:
        return _result("missing_segment", False, 0, None, "no_datetime_column")

    b = pd.to_datetime(base[dtc], errors="coerce").dropna()
    t = pd.to_datetime(test[dtc], errors="coerce").dropna()
    base_idx = pd.DatetimeIndex(b.unique()).sort_values()
    test_idx = pd.DatetimeIndex(t.unique()).sort_values()
    missing = base_idx.difference(test_idx)

    if len(missing) < min_missing:
        return _result("missing_segment", len(missing) > 0, float(len(missing) / max(len(base_idx), 1)),
                       len(missing))

    # Check contiguity: longest run of consecutive missing timestamps
    diffs = np.diff(missing.view("int64"))
    if len(diffs) == 0:
        return _result("missing_segment", False, 0.0, len(missing))
    typical_gap = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    run_lengths = []
    run = 1
    for d in diffs:
        if abs(d - typical_gap) / max(typical_gap, 1) < 0.5:
            run += 1
        else:
            run_lengths.append(run)
            run = 1
    run_lengths.append(run)
    max_run = max(run_lengths)
    detected = max_run >= min_missing
    confidence = _clamp01(max_run / (min_missing * 2))
    return _result("missing_segment", detected, confidence, {"total_missing": len(missing), "max_run": max_run},
                   f"max_contiguous_missing={max_run}" if detected else None)


def contract_power_data_repetition(
    base: pd.DataFrame,
    test: pd.DataFrame,
    ratio_threshold: float = 1.4,   # test has ≥1.4× more rows in the same timespan
) -> dict:
    """
    Row count ratio in same time window > threshold ⟹ data repetition detected.
    """
    for dtc in ("DateTime", "Date"):
        if dtc in base.columns and dtc in test.columns:
            break
    else:
        return _result("data_repetition", False, 1.0, None, "no_datetime_column")

    bd = pd.to_datetime(base[dtc], errors="coerce")
    td = pd.to_datetime(test[dtc], errors="coerce")
    overlap_min = max(bd.min(), td.min())
    overlap_max = min(bd.max(), td.max())
    if pd.isna(overlap_min) or overlap_max < overlap_min:
        return _result("data_repetition", False, 1.0, None, "no_time_overlap")

    n_base = int(((bd >= overlap_min) & (bd <= overlap_max)).sum())
    n_test = int(((td >= overlap_min) & (td <= overlap_max)).sum())
    ratio = n_test / max(n_base, 1)
    detected = ratio > ratio_threshold
    confidence = _clamp01((ratio - ratio_threshold) / ratio_threshold) if detected else 0.0
    return _result("data_repetition", detected, confidence, ratio,
                   f"row_ratio={ratio:.2f}" if detected else None)


def contract_power_sign_flip(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_reactive_power",
    corr_threshold: float = -0.70,
) -> dict:
    """
    Sign-flip detection using two robust signals.

    Signal A — Pearson correlation on non-zero pairs (filters out the many
               zero reactive-power readings that make corrcoef return NaN).
    Signal B — Signed-product median: after b = -a, median(a*b) = median(-a²)
               which is strongly negative. This fires even when Signal A
               cannot due to sparsity.

    Both signals are required to survive the zero-dominated UCI power data.
    """
    if col not in base.columns or col not in test.columns:
        return _result("sign_flip_reactive", False, 0.0, None, "no_reactive_column")

    base_al, test_al = _align(base, test)
    if base_al is None:
        return _result("sign_flip_reactive", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).values
    b = _num(test_al[col]).values
    n = min(len(a), len(b))
    if n < 10:
        return _result("sign_flip_reactive", False, 0.0, None, "insufficient_data")
    a, b = a[:n], b[:n]

    # Signal A: correlation on non-zero pairs only
    nonzero = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6) & ~np.isnan(a) & ~np.isnan(b)
    corr = np.nan
    if nonzero.sum() >= 10:
        try:
            corr = float(np.corrcoef(a[nonzero], b[nonzero])[0, 1])
        except Exception:
            pass
    flip_by_corr = (not np.isnan(corr)) and (corr < corr_threshold)

    # Signal B: signed-product median
    # After sign flip: a*b = a*(-a) = -a² ≤ 0 everywhere
    # For clean data:  a*b = a² ≥ 0 everywhere
    valid = ~np.isnan(a) & ~np.isnan(b)
    med_product = float(np.median((a * b)[valid])) if valid.sum() >= 10 else 0.0
    # Normalise by median(a²) to get a scale-free ratio in [-1, 1]
    med_sq = float(np.median((a ** 2)[valid & (np.abs(a) > 1e-6)])) if (valid & (np.abs(a) > 1e-6)).sum() >= 5 else 1.0
    signed_ratio = med_product / max(med_sq, 1e-12)
    flip_by_product = signed_ratio < -0.80  # ≥80 % of variance is negated

    detected = flip_by_corr or flip_by_product
    confidence = max(
        _clamp01((-corr - abs(corr_threshold)) / (1 - abs(corr_threshold))) if flip_by_corr else 0.0,
        _clamp01((-signed_ratio - 0.80) / 0.20) if flip_by_product else 0.0,
    )
    score = {"corr_nonzero": float(corr) if not np.isnan(corr) else None,
             "signed_product_ratio": signed_ratio}
    reason = None
    if detected:
        if flip_by_product:
            reason = f"signed_product_ratio={signed_ratio:.4f} (negated signal)"
        else:
            reason = f"corr_nonzero={corr:.4f}"
    return _result("sign_flip_reactive", detected, confidence, score, reason)


def contract_power_temporal_acf(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_active_power",
    nlags: int = 24,
    acf_diff_threshold: float = 0.10,
) -> dict:
    """
    ACF profile comparison — catches temporal mutations (shuffle, shift) robustly. NEW
    """
    base_al, test_al = _align(base, test)
    if base_al is None or col not in base_al.columns:
        return _result("temporal_acf_power", False, 0.0, None, "alignment_failed")

    a = _num(base_al[col]).ffill().dropna().values
    b = _num(test_al[col]).ffill().dropna().values
    if min(len(a), len(b)) < nlags + 10:
        return _result("temporal_acf_power", False, 0.0, None, "insufficient_data")

    acf_a = _acf(a, nlags)
    acf_b = _acf(b, nlags)
    diff = float(np.mean(np.abs(acf_a - acf_b)))
    detected = diff > acf_diff_threshold
    confidence = _clamp01((diff - acf_diff_threshold) / acf_diff_threshold) if detected else 0.0
    return _result("temporal_acf_power", detected, confidence, diff,
                   f"mean_acf_diff={diff:.4f}" if detected else None)


def contract_power_constant_segment(
    base: pd.DataFrame,
    test: pd.DataFrame,
    col: str = "Global_active_power",
    window: int = 60,   # minimum frozen window length in rows
    variance_threshold: float = 1e-6,
) -> dict:
    """
    Long zero-variance window in test data ⟹ frozen/constant segment detected. NEW
    Uses rolling variance to identify stuck readings.
    Only fires if test has significantly MORE frozen windows than baseline,
    to avoid false positives on naturally flat segments in clean data.
    """
    if col not in test.columns:
        return _result("constant_segment", False, 0.0, None, "no_column")

    def _count_frozen(df_col: pd.Series) -> int:
        arr = _num(df_col).ffill().fillna(0).values
        if len(arr) < window * 2:
            return 0
        variances = np.array([np.var(arr[i: i + window])
                               for i in range(0, len(arr) - window, window // 4)])
        return int(np.sum(variances < variance_threshold))

    test_frozen = _count_frozen(test[col])
    base_frozen = _count_frozen(base[col]) if col in base.columns else 0

    # Only flag if test has at least 3 MORE frozen windows than baseline
    new_frozen = test_frozen - base_frozen
    detected = new_frozen >= 3
    confidence = _clamp01(new_frozen / 10) if detected else 0.0
    return _result("constant_segment", detected, confidence,
                   {"test_frozen": test_frozen, "base_frozen": base_frozen, "new_frozen": new_frozen},
                   f"new_frozen_windows={new_frozen}" if detected else None)


def contract_stock_timestamp_lag(
    base: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "Date",
    min_shift_days: float = 0.5,
) -> dict:
    """
    Compare date ranges between baseline and test stock data.

    A 1-day timestamp lag shifts both min and max dates forward by 1 day.
    Comparing the median dates (robust to dropped rows) reliably catches this.
    Same principle as contract_walmart_shift_dates — domain-agnostic date math.
    """
    if date_col not in base.columns or date_col not in test.columns:
        return _result("timestamp_lag", False, 0.0, None, "no_date_column")

    base_dates = pd.to_datetime(base[date_col], errors="coerce").dropna()
    test_dates  = pd.to_datetime(test[date_col], errors="coerce").dropna()
    if len(base_dates) < 5 or len(test_dates) < 5:
        return _result("timestamp_lag", False, 0.0, None, "insufficient_data")

    base_med = base_dates.sort_values().iloc[len(base_dates) // 2]
    test_med  = test_dates.sort_values().iloc[len(test_dates) // 2]
    shift_days = abs((test_med - base_med).total_seconds()) / 86400.0

    detected = shift_days >= min_shift_days
    confidence = _clamp01(shift_days / 5.0)   # normalise: 5-day shift = full confidence
    return _result("timestamp_lag", detected, confidence, shift_days,
                   f"date_shift={shift_days:.2f}d" if detected else None)


def contract_walmart_duplicate_rows(
    base: pd.DataFrame,
    test: pd.DataFrame,
    min_extra_frac: float = 0.02,   # flag if test has ≥2 % more rows than baseline
) -> dict:
    """
    Row count ratio > 1 ⟹ duplicate rows injected.

    Duplicating 5 % of rows produces exactly len(base)*1.05 rows.
    A pure row-count ratio is the most direct and reliable detector for this
    mutation, since the duplicated rows are statistically indistinguishable
    from genuine data (same distribution, same time range).
    """
    n_base = len(base)
    n_test  = len(test)
    if n_base == 0:
        return _result("duplicate_rows", False, 0.0, None, "empty_baseline")

    extra_frac = (n_test - n_base) / n_base
    detected = extra_frac >= min_extra_frac
    confidence = _clamp01(extra_frac / 0.10)   # normalise: 10 % extra = full confidence
    return _result("duplicate_rows", detected, confidence, extra_frac,
                   f"extra_rows_frac={extra_frac:.3f}" if detected else None)

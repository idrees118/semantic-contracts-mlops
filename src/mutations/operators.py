"""
src/mutations/operators.py
==========================
All 28 semantic mutation operators across 5 categories:
  Scaling (6), Temporal (6), Structural (9), Quality (6), Aggregation (1)

Every operator is a pure function:
    mutate_fn(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame

The rng parameter is seeded externally so that multi-seed evaluation is
reproducible and consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dt_col(df: pd.DataFrame) -> str | None:
    for c in ("DateTime", "Date", "date", "datetime", "Timestamp"):
        if c in df.columns:
            return c
    return None


# ===========================================================================
# WALMART OPERATORS
# ===========================================================================

# --- Temporal ---

def walmart_shift_dates_neg7(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shift all dates backward by 7 days (temporal)."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce") - pd.Timedelta(days=7)
    return d.sort_values("Date").reset_index(drop=True)


def walmart_shift_dates_pos7(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shift all dates forward by 7 days (temporal)."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce") + pd.Timedelta(days=7)
    return d.sort_values("Date").reset_index(drop=True)


# --- Scaling ---

def walmart_scale_sales_09(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply Weekly_Sales by 0.9 — subtle 10 % revenue undercount (scaling)."""
    d = df.copy()
    d["Weekly_Sales"] = _numeric(d["Weekly_Sales"]) * 0.9
    return d


def walmart_scale_temperature_2x(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply Temperature by 2 — unit confusion °F→°C gone wrong (scaling). NEW"""
    d = df.copy()
    if "Temperature" in d.columns:
        d["Temperature"] = _numeric(d["Temperature"]) * 2.0
    return d


# --- Structural ---

def walmart_toggle_holiday(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Flip all Holiday_Flag values 0↔1 (structural)."""
    d = df.copy()
    for c in ("Holiday_Flag", "IsHoliday", "Holiday"):
        if c in d.columns:
            d[c] = d[c].apply(lambda x: 0 if int(x) else 1)
            break
    return d


def walmart_missing_recent_10pct(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Drop the most recent 10 % of rows (structural — truncated pipeline)."""
    d = df.copy()
    n_drop = max(1, int(len(d) * 0.10))
    return d.iloc[: len(d) - n_drop].reset_index(drop=True)


def walmart_missing_stores_5(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Drop all rows for up to 5 randomly chosen stores (structural). NEW
    Always keeps at least one store to prevent empty DataFrames."""
    d = df.copy()
    if "Store" not in d.columns:
        return d
    stores = sorted(d["Store"].dropna().unique().tolist())
    # Drop up to 5 stores, but always keep at least 1
    n_drop = min(5, max(0, len(stores) - 1))
    if n_drop == 0:
        return d
    idx = rng.choice(len(stores), size=n_drop, replace=False)
    chosen = [stores[i] for i in idx]
    return d[~d["Store"].isin(chosen)].reset_index(drop=True)


# --- Quality ---

def walmart_duplicate_rows_5pct(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Duplicate a random 5 % of rows (quality — ETL copy-paste bug)."""
    d = df.copy()
    n_dup = max(1, int(len(d) * 0.05))
    idx = rng.choice(d.index, size=n_dup, replace=False)
    return pd.concat([d, d.loc[idx]], ignore_index=True)


# --- Aggregation ---

def walmart_wrong_aggregation_x7(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply Weekly_Sales by 7 — daily→weekly aggregation bug (aggregation)."""
    d = df.copy()
    d["Weekly_Sales"] = _numeric(d["Weekly_Sales"]) * 7
    return d


# ===========================================================================
# STOCK OPERATORS
# ===========================================================================

# --- Quality ---

def stock_noise_001(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add Gaussian noise σ=0.01·std to all price columns (quality)."""
    d = df.copy()
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in d.columns]
    for c in price_cols:
        arr = _numeric(d[c])
        sigma = 0.01 * float(arr.std(skipna=True))
        d[c] = arr + rng.normal(0, sigma if sigma > 0 else 1e-4, size=len(arr))
    return d


def stock_zero_volume_30pct(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Set Volume to 0 for 30 % of rows — sparse volume data bug (quality). NEW"""
    d = df.copy()
    if "Volume" not in d.columns:
        return d
    idx = rng.choice(d.index, size=int(len(d) * 0.30), replace=False)
    d.loc[idx, "Volume"] = 0
    return d


# --- Structural ---

def stock_drop_5pct(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Drop 5 % of rows randomly (structural)."""
    d = df.copy()
    n = max(1, int(len(d) * 0.05))
    idx = rng.choice(d.index, size=n, replace=False)
    return d.drop(index=idx).reset_index(drop=True)


def stock_column_swap_open_close(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Swap Open and Close columns (structural — column ordering bug)."""
    d = df.copy()
    if "Open" in d.columns and "Close" in d.columns:
        d["Open"], d["Close"] = d["Close"].copy(), d["Open"].copy()
    return d


# --- Scaling ---

def stock_shift_price_05(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add $0.50 to all price columns (scaling — systematic pricing bias)."""
    d = df.copy()
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        if c in d.columns:
            d[c] = _numeric(d[c]) + 0.5
    return d


def stock_unit_conversion_x10(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply all prices by 10 — unit confusion (cents → dollars bug) (scaling)."""
    d = df.copy()
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        if c in d.columns:
            d[c] = _numeric(d[c]) * 10.0
    return d


# --- Temporal ---

def stock_timestamp_lag_1day(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shift all timestamps forward by 1 trading day (temporal)."""
    d = df.copy()
    dc = _dt_col(d)
    if dc:
        d[dc] = pd.to_datetime(d[dc], errors="coerce") + pd.Timedelta(days=1)
    return d


def stock_reverse_series(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reverse the entire time series — as-of dates completely wrong (temporal). NEW"""
    d = df.copy()
    dc = _dt_col(d)
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in d.columns]
    if dc:
        d = d.sort_values(dc)
        # Reverse only the price values, keep timestamps in place
        for c in price_cols:
            d[c] = d[c].values[::-1]
    return d.reset_index(drop=True)


# ===========================================================================
# POWER OPERATORS
# ===========================================================================

# --- Scaling ---

def power_scale_active_11(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply Global_active_power by 1.1 — calibration drift (scaling)."""
    d = df.copy()
    d["Global_active_power"] = _numeric(d["Global_active_power"]) * 1.1
    return d


def power_scale_voltage_05(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply Voltage by 0.5 — sensor gain error (scaling). NEW"""
    d = df.copy()
    if "Voltage" in d.columns:
        d["Voltage"] = _numeric(d["Voltage"]) * 0.5
    return d


# --- Temporal ---

def power_shuffle_within_day(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle records within each day (temporal — intra-day ordering lost)."""
    d = df.copy()
    dc = _dt_col(d)
    if dc is None:
        return d
    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d["__day"] = d[dc].dt.date
    parts = []
    for _, g in d.groupby("__day"):
        parts.append(g.sample(frac=1, random_state=int(rng.integers(0, 2**31))))
    return pd.concat(parts).drop(columns=["__day"]).reset_index(drop=True)


def power_shift_by_1week(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shift all timestamps forward by exactly 7 days (temporal). NEW"""
    d = df.copy()
    dc = _dt_col(d)
    if dc:
        d[dc] = pd.to_datetime(d[dc], errors="coerce") + pd.Timedelta(days=7)
    return d


# --- Structural ---

def power_column_swap_voltage_intensity(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Swap Voltage and Global_intensity columns (structural)."""
    d = df.copy()
    if "Voltage" in d.columns and "Global_intensity" in d.columns:
        d["Voltage"], d["Global_intensity"] = d["Global_intensity"].copy(), d["Voltage"].copy()
    return d


def power_missing_6hour_segment(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Drop a contiguous 6-hour block of data (structural — outage / ETL gap)."""
    d = df.copy()
    dc = _dt_col(d)
    if dc is None:
        return d
    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d = d.sort_values(dc)
    # Choose a random start in the middle 50 % of the series
    n = len(d)
    start_idx = int(rng.integers(n // 4, 3 * n // 4))
    # 6 hours = 360 minutes at 1-min frequency
    end_idx = min(start_idx + 360, n)
    mask = np.ones(n, dtype=bool)
    mask[start_idx:end_idx] = False
    return d[mask].reset_index(drop=True)


def power_data_repetition_5x(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tile the dataset 5× — pipeline loop bug repeating data (structural)."""
    return pd.concat([df] * 5, ignore_index=True)


def power_sign_flip_reactive(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Flip the sign of Global_reactive_power (structural). NEW"""
    d = df.copy()
    if "Global_reactive_power" in d.columns:
        d["Global_reactive_power"] = -_numeric(d["Global_reactive_power"])
    return d


# --- Quality ---

def power_inject_spikes_100_3(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject 100 spikes at ±3 σ into Global_active_power (quality)."""
    d = df.copy()
    col = "Global_active_power"
    if col not in d.columns:
        return d
    arr = _numeric(d[col]).fillna(0).values
    std = float(np.std(arr)) or 1.0
    idx = rng.choice(len(arr), size=100, replace=False)
    d.loc[idx, col] = arr[idx] + 3.0 * std
    return d


def power_extreme_outliers_50(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject 50 extreme outliers at 10 σ (quality — sensor malfunction sim)."""
    d = df.copy()
    col = "Global_active_power"
    if col not in d.columns:
        return d
    arr = _numeric(d[col]).fillna(0).values
    std = float(np.std(arr)) or 1.0
    idx = rng.choice(len(arr), size=50, replace=False)
    d.loc[idx, col] = arr[idx] + 10.0 * std
    return d


def power_constant_segment_2h(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace a 2-hour window with the column mean — sensor freeze (quality). NEW"""
    d = df.copy()
    col = "Global_active_power"
    if col not in d.columns:
        return d
    arr = _numeric(d[col]).values
    mean_val = float(np.nanmean(arr))
    n = len(d)
    start = int(rng.integers(0, max(1, n - 120)))
    end = min(start + 120, n)
    d.loc[d.index[start:end], col] = mean_val
    return d

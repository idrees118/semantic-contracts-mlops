#!/usr/bin/env python3
"""
scripts/prepare_datasets.py
=============================
Reads the three raw datasets from data/processed/ (wherever you placed them),
standardises column names, parses dates, drops nulls, and writes out the
clean CSVs that the experiment expects.

Run once before running run_experiment.py:

    python scripts/prepare_datasets.py

Input files (place these in data/processed/):
    AAPL_5year.csv
    household_power_consumption_1M.csv
    walmart.csv   (or Walmart.csv)

Output files (written to data/processed/):
    stock_clean.csv
    power_clean.csv
    walmart_clean.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find(candidates: list[str]) -> Path | None:
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


def _info(df: pd.DataFrame, label: str) -> None:
    print(f"  [{label}] {len(df):,} rows  |  columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# STOCK  (AAPL_5year.csv)
# ---------------------------------------------------------------------------

def prepare_stock() -> None:
    path = _find(["AAPL_5year.csv", "aapl_5year.csv", "AAPL.csv"])
    if path is None:
        print("[SKIP] Stock: AAPL_5year.csv not found in data/processed/")
        return

    df = pd.read_csv(path)
    print(f"\nStock raw columns: {list(df.columns)}")

    # Normalise column names — Yahoo Finance uses Title Case
    rename = {}
    for col in df.columns:
        c = col.strip()
        cl = c.lower()
        if cl in ("date", "datetime", "timestamp"):
            rename[col] = "Date"
        elif cl == "open":
            rename[col] = "Open"
        elif cl == "high":
            rename[col] = "High"
        elif cl == "low":
            rename[col] = "Low"
        elif cl == "close":
            rename[col] = "Close"
        elif cl in ("adj close", "adj_close", "adjclose", "adjusted close"):
            rename[col] = "Adj Close"
        elif cl == "volume":
            rename[col] = "Volume"
    df = df.rename(columns=rename)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # If Adj Close not present, copy from Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Keep only needed columns (drop any extras)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if c in df.columns]
    df = df[keep]

    # Convert price columns to numeric, drop rows with all-NaN prices
    price_cols = [c for c in keep if c != "Date"]
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    out = DATA_DIR / "stock_clean.csv"
    df.to_csv(out, index=False)
    _info(df, "stock_clean.csv")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# POWER  (household_power_consumption_1M.csv)
# ---------------------------------------------------------------------------

def prepare_power() -> None:
    candidates = [
        "household_power_consumption_1M.csv",
        "household_power_consumption.csv",
        "power_consumption.csv",
    ]
    path = _find(candidates)
    if path is None:
        print("[SKIP] Power: household_power_consumption_1M.csv not found in data/processed/")
        return

    # UCI dataset is semicolon-separated; try both
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        if len(df.columns) < 3:
            df = pd.read_csv(path, sep=",", low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)

    print(f"\nPower raw columns: {list(df.columns)}")

    # UCI format has separate Date and Time columns
    rename = {}
    for col in df.columns:
        c = col.strip()
        cl = c.lower().replace(" ", "_")
        if cl == "date":
            rename[col] = "RawDate"
        elif cl == "time":
            rename[col] = "RawTime"
        elif cl in ("datetime", "date_time"):
            rename[col] = "DateTime"
        elif cl == "global_active_power":
            rename[col] = "Global_active_power"
        elif cl == "global_reactive_power":
            rename[col] = "Global_reactive_power"
        elif cl == "voltage":
            rename[col] = "Voltage"
        elif cl in ("global_intensity", "global_intensity_"):
            rename[col] = "Global_intensity"
        elif cl in ("sub_metering_1", "sub_metering1"):
            rename[col] = "Sub_metering_1"
        elif cl in ("sub_metering_2", "sub_metering2"):
            rename[col] = "Sub_metering_2"
        elif cl in ("sub_metering_3", "sub_metering3"):
            rename[col] = "Sub_metering_3"
    df = df.rename(columns=rename)

    # Build DateTime column
    if "DateTime" not in df.columns:
        if "RawDate" in df.columns and "RawTime" in df.columns:
            df["DateTime"] = pd.to_datetime(
                df["RawDate"].astype(str) + " " + df["RawTime"].astype(str),
                dayfirst=True, errors="coerce"
            )
            df = df.drop(columns=["RawDate", "RawTime"])
        else:
            print("  [WARN] Cannot find Date/Time columns — trying first column as DateTime")
            df = df.rename(columns={df.columns[0]: "DateTime"})
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    # Replace "?" (UCI missing marker) with NaN and convert numerics
    numeric_cols = ["Global_active_power", "Global_reactive_power",
                    "Voltage", "Global_intensity",
                    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("?", np.nan), errors="coerce")

    # Drop rows missing the key measurement column
    if "Global_active_power" in df.columns:
        df = df.dropna(subset=["Global_active_power"])

    # Keep only needed columns
    keep = ["DateTime"] + [c for c in numeric_cols if c in df.columns]
    df = df[keep].reset_index(drop=True)

    out = DATA_DIR / "power_clean.csv"
    df.to_csv(out, index=False)
    _info(df, "power_clean.csv")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# WALMART  (walmart.csv)
# ---------------------------------------------------------------------------

def prepare_walmart() -> None:
    path = _find(["walmart.csv", "Walmart.csv", "walmart_clean_raw.csv", "train.csv"])
    if path is None:
        print("[SKIP] Walmart: walmart.csv not found in data/processed/")
        return

    df = pd.read_csv(path)
    print(f"\nWalmart raw columns: {list(df.columns)}")

    rename = {}
    for col in df.columns:
        c = col.strip()
        cl = c.lower().replace(" ", "_")
        if cl in ("date",):
            rename[col] = "Date"
        elif cl in ("store",):
            rename[col] = "Store"
        elif cl in ("weekly_sales", "sales", "weeklysales"):
            rename[col] = "Weekly_Sales"
        elif cl in ("holiday_flag", "isholiday", "is_holiday", "holiday"):
            rename[col] = "Holiday_Flag"
        elif cl in ("temperature", "temp"):
            rename[col] = "Temperature"
        elif cl in ("fuel_price", "fuelprice"):
            rename[col] = "Fuel_Price"
        elif cl in ("cpi",):
            rename[col] = "CPI"
        elif cl in ("unemployment",):
            rename[col] = "Unemployment"
        elif cl in ("dept", "department"):
            rename[col] = "Dept"
        elif cl in ("type",):
            rename[col] = "Type"
        elif cl in ("size",):
            rename[col] = "Size"
        elif cl in ("markdowna", "markdownb", "markdownc", "markdownd", "markdowne",
                    "markdown1", "markdown2", "markdown3", "markdown4", "markdown5"):
            rename[col] = col  # keep as-is
    df = df.rename(columns=rename)

    # Parse date — Walmart uses DD/MM/YYYY or YYYY-MM-DD
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    else:
        print("  [ERROR] No Date column found in Walmart file!")
        return

    # Standardise Holiday_Flag: convert True/False to 1/0
    if "Holiday_Flag" in df.columns:
        hf = df["Holiday_Flag"]
        if hf.dtype == object or hf.dtype == bool:
            df["Holiday_Flag"] = hf.map(
                lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
            )
        df["Holiday_Flag"] = df["Holiday_Flag"].astype(int)

    # Convert sales to numeric
    if "Weekly_Sales" in df.columns:
        df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
        df = df.dropna(subset=["Weekly_Sales"])

    # Keep useful columns
    keep_candidates = [
        "Date", "Store", "Dept", "Weekly_Sales",
        "Holiday_Flag", "Temperature", "Fuel_Price",
        "CPI", "Unemployment", "Type", "Size",
    ]
    keep = [c for c in keep_candidates if c in df.columns]
    df = df[keep].reset_index(drop=True)

    out = DATA_DIR / "walmart_clean.csv"
    df.to_csv(out, index=False)
    _info(df, "walmart_clean.csv")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Looking for raw datasets in: {DATA_DIR}")
    print(f"Files found: {[f.name for f in DATA_DIR.iterdir() if f.is_file()]}")

    prepare_stock()
    prepare_walmart()
    prepare_power()

    print("\nDone. Clean files in data/processed/:")
    for f in sorted(DATA_DIR.glob("*_clean.csv")):
        rows = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:<30} {rows:>10,} rows")


if __name__ == "__main__":
    main()

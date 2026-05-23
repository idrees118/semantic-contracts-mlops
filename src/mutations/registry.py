"""
src/mutations/registry.py
==========================
Central registry of all 28 semantic mutations.

Each entry is a MutationSpec dataclass containing:
  - name:       unique identifier used in filenames and result tables
  - dataset:    which dataset it targets  ("walmart" | "stock" | "power")
  - category:   taxonomy category         ("scaling" | "temporal" | "structural" |
                                           "quality"  | "aggregation")
  - fn:         the operator callable     (df, rng) -> df
  - description: one-line human description

Usage
-----
from src.mutations.registry import MUTATIONS, get_mutations_for_dataset

for spec in get_mutations_for_dataset("walmart"):
    mutated_df = spec.fn(clean_df, rng)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from src.mutations.operators import (
    # Walmart
    walmart_shift_dates_neg7,
    walmart_shift_dates_pos7,
    walmart_scale_sales_09,
    walmart_scale_temperature_2x,
    walmart_toggle_holiday,
    walmart_missing_recent_10pct,
    walmart_missing_stores_5,
    walmart_duplicate_rows_5pct,
    walmart_wrong_aggregation_x7,
    # Stock
    stock_noise_001,
    stock_zero_volume_30pct,
    stock_drop_5pct,
    stock_column_swap_open_close,
    stock_shift_price_05,
    stock_unit_conversion_x10,
    stock_timestamp_lag_1day,
    stock_reverse_series,
    # Power
    power_scale_active_11,
    power_scale_voltage_05,
    power_shuffle_within_day,
    power_shift_by_1week,
    power_column_swap_voltage_intensity,
    power_missing_6hour_segment,
    power_data_repetition_5x,
    power_sign_flip_reactive,
    power_inject_spikes_100_3,
    power_extreme_outliers_50,
    power_constant_segment_2h,
)


@dataclass(frozen=True)
class MutationSpec:
    name: str
    dataset: str
    category: str
    fn: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]
    description: str


MUTATIONS: list[MutationSpec] = [
    # ------------------------------------------------------------------ Walmart
    MutationSpec(
        "walmart__shift_dates_-7", "walmart", "temporal",
        walmart_shift_dates_neg7,
        "Shift all dates backward 7 days — simulates a weekly off-by-one ETL bug.",
    ),
    MutationSpec(
        "walmart__shift_dates_7", "walmart", "temporal",
        walmart_shift_dates_pos7,
        "Shift all dates forward 7 days — future-dated records in the pipeline.",
    ),
    MutationSpec(
        "walmart__scale_sales_0.9", "walmart", "scaling",
        walmart_scale_sales_09,
        "Multiply Weekly_Sales by 0.9 — subtle 10 % systematic undercount.",
    ),
    MutationSpec(
        "walmart__scale_temperature_2x", "walmart", "scaling",
        walmart_scale_temperature_2x,
        "Multiply Temperature by 2 — unit confusion (°F reported as 2×°F).",
    ),
    MutationSpec(
        "walmart__toggle_holiday", "walmart", "structural",
        walmart_toggle_holiday,
        "Flip Holiday_Flag 0↔1 — inverts promotion/holiday logic.",
    ),
    MutationSpec(
        "walmart__missing_recent_10pct", "walmart", "structural",
        walmart_missing_recent_10pct,
        "Drop the most recent 10 % of rows — truncated pipeline delivery.",
    ),
    MutationSpec(
        "walmart__missing_stores_5", "walmart", "structural",
        walmart_missing_stores_5,
        "Drop all rows for 5 stores — partial store-level data omission.",
    ),
    MutationSpec(
        "walmart__duplicate_rows_5pct", "walmart", "quality",
        walmart_duplicate_rows_5pct,
        "Duplicate 5 % of rows — ETL re-processing / double-append bug.",
    ),
    MutationSpec(
        "walmart__wrong_aggregation_x7", "walmart", "aggregation",
        walmart_wrong_aggregation_x7,
        "Multiply Weekly_Sales by 7 — daily-to-weekly aggregation applied twice.",
    ),
    # ------------------------------------------------------------------ Stock
    MutationSpec(
        "stock__noise_0.01", "stock", "quality",
        stock_noise_001,
        "Add Gaussian noise σ=0.01·std to all price columns — feed rounding noise.",
    ),
    MutationSpec(
        "stock__zero_volume_30pct", "stock", "quality",
        stock_zero_volume_30pct,
        "Zero out Volume for 30 % of rows — sparse volume data after join bug.",
    ),
    MutationSpec(
        "stock__drop_5pct", "stock", "structural",
        stock_drop_5pct,
        "Drop 5 % of rows randomly — missing trading days / partial delivery.",
    ),
    MutationSpec(
        "stock__column_swap_open_close", "stock", "structural",
        stock_column_swap_open_close,
        "Swap Open and Close — column ordering error in schema migration.",
    ),
    MutationSpec(
        "stock__shift_price_0.5", "stock", "scaling",
        stock_shift_price_05,
        "Add $0.50 to all prices — systematic pricing bias / fee misapplied.",
    ),
    MutationSpec(
        "stock__unit_conversion_x10", "stock", "scaling",
        stock_unit_conversion_x10,
        "Multiply prices by 10 — cents-to-dollars conversion applied twice.",
    ),
    MutationSpec(
        "stock__timestamp_lag_1day", "stock", "temporal",
        stock_timestamp_lag_1day,
        "Lag all timestamps by 1 trading day — T+1 settlement date confusion.",
    ),
    MutationSpec(
        "stock__reverse_series", "stock", "temporal",
        stock_reverse_series,
        "Reverse price series — as-of date completely inverted.",
    ),
    # ------------------------------------------------------------------ Power
    MutationSpec(
        "power__scale_active_1.1", "power", "scaling",
        power_scale_active_11,
        "Multiply Global_active_power by 1.1 — meter calibration drift.",
    ),
    MutationSpec(
        "power__scale_voltage_0.5", "power", "scaling",
        power_scale_voltage_05,
        "Multiply Voltage by 0.5 — sensor gain halved after firmware update.",
    ),
    MutationSpec(
        "power__shuffle_within_day", "power", "temporal",
        power_shuffle_within_day,
        "Shuffle records within each calendar day — intra-day ordering destroyed.",
    ),
    MutationSpec(
        "power__shift_by_1week", "power", "temporal",
        power_shift_by_1week,
        "Shift all timestamps forward 7 days — weekly cron job date arithmetic bug.",
    ),
    MutationSpec(
        "power__column_swap_voltage_intensity", "power", "structural",
        power_column_swap_voltage_intensity,
        "Swap Voltage and Global_intensity — schema mismatch after column rename.",
    ),
    MutationSpec(
        "power__missing_6hour_segment", "power", "structural",
        power_missing_6hour_segment,
        "Delete a contiguous 6-hour block — sensor outage / ETL gap.",
    ),
    MutationSpec(
        "power__data_repetition_5x", "power", "structural",
        power_data_repetition_5x,
        "Tile data 5× — pipeline loop bug causing data repeated 5 times.",
    ),
    MutationSpec(
        "power__sign_flip_reactive", "power", "structural",
        power_sign_flip_reactive,
        "Negate Global_reactive_power — sign convention error in transformer.",
    ),
    MutationSpec(
        "power__inject_spikes_100_3.0", "power", "quality",
        power_inject_spikes_100_3,
        "Inject 100 spikes at 3σ — transient sensor malfunction events.",
    ),
    MutationSpec(
        "power__extreme_outliers_50", "power", "quality",
        power_extreme_outliers_50,
        "Inject 50 outliers at 10σ — severe sensor error or transmission fault.",
    ),
    MutationSpec(
        "power__constant_segment_2h", "power", "quality",
        power_constant_segment_2h,
        "Freeze a 2-hour window at mean value — sensor lock-up / stuck reading.",
    ),
]

# Convenience accessors
MUTATION_NAMES: list[str] = [m.name for m in MUTATIONS]
CATEGORIES: list[str] = sorted({m.category for m in MUTATIONS})


def get_mutations_for_dataset(dataset: str) -> list[MutationSpec]:
    return [m for m in MUTATIONS if m.dataset == dataset.lower()]


def get_mutations_by_category(category: str) -> list[MutationSpec]:
    return [m for m in MUTATIONS if m.category == category.lower()]


def get_mutation(name: str) -> MutationSpec:
    for m in MUTATIONS:
        if m.name == name:
            return m
    raise KeyError(f"Unknown mutation: {name!r}")


def print_taxonomy() -> None:
    """Pretty-print the full mutation taxonomy."""
    from collections import defaultdict
    by_cat: dict[str, list[MutationSpec]] = defaultdict(list)
    for m in MUTATIONS:
        by_cat[m.category].append(m)
    total = 0
    for cat in sorted(by_cat):
        items = by_cat[cat]
        print(f"\n[{cat.upper()}]  ({len(items)} mutations)")
        for m in items:
            print(f"  {m.name:<45}  {m.description[:70]}")
            total += 1
    print(f"\nTotal: {total} mutations across {len(by_cat)} categories")


if __name__ == "__main__":
    print_taxonomy()

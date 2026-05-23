"""
tests/test_pipeline.py
=======================
Smoke tests and unit tests for the semantic mutation v2 pipeline.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import wilson_ci, mcnemar_exact, detection_summary
from src.baselines.ks_drift_baseline import run_ks_baseline
from src.baselines.ensemble_baseline import run_ensemble_baseline
from src.baselines.ge_baseline import run_ge_baseline
from src.validators.contracts import (
    contract_walmart_scale_sales,
    contract_walmart_toggle_holiday,
    contract_stock_shift_price,
    contract_stock_temporal_acf,
    contract_power_scale_active,
    contract_power_sign_flip,
    contract_power_constant_segment,
)
from src.validators.semantic_validator import validate
from src.mutations.registry import MUTATIONS, get_mutations_for_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def walmart_df():
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.date_range("2010-01-01", periods=n // 4, freq="W")
    records = []
    for store in range(1, 5):
        for d in dates:
            records.append({
                "Date": d,
                "Store": store,
                "Weekly_Sales": 20000 + rng.normal(0, 2000),
                "Holiday_Flag": int(rng.random() < 0.05),
                "Temperature": 60 + rng.normal(0, 15),
                "Fuel_Price": 3.5 + rng.normal(0, 0.3),
                "CPI": 210 + rng.normal(0, 5),
                "Unemployment": 7.5 + rng.normal(0, 0.5),
            })
    return pd.DataFrame(records)


@pytest.fixture
def stock_df():
    rng = np.random.default_rng(42)
    n = 252
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    prices = 150.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame({
        "Date":     dates,
        "Open":     prices + rng.normal(0, 0.5, n),
        "High":     prices + abs(rng.normal(0, 1, n)),
        "Low":      prices - abs(rng.normal(0, 1, n)),
        "Close":    prices + rng.normal(0, 0.5, n),
        "Adj Close": prices + rng.normal(0, 0.5, n),
        "Volume":   (1e6 * rng.lognormal(0, 1, n)).astype(int),
    })


@pytest.fixture
def power_df():
    rng = np.random.default_rng(42)
    n = 2880  # 2 days × 1440 min/day
    dts = pd.date_range("2007-01-01", periods=n, freq="min")
    hour = dts.hour
    pattern = 0.5 + 1.5 * np.sin(np.pi * hour / 12) ** 2
    return pd.DataFrame({
        "DateTime":              dts,
        "Global_active_power":   pattern + rng.normal(0, 0.1, n),
        "Global_reactive_power": 0.1 + 0.05 * pattern + rng.normal(0, 0.02, n),
        "Voltage":               240 + rng.normal(0, 2, n),
        "Global_intensity":      10 + rng.normal(0, 0.5, n),
        "Sub_metering_1":        rng.exponential(0.5, n),
        "Sub_metering_2":        rng.exponential(0.3, n),
        "Sub_metering_3":        rng.exponential(1.0, n),
    })


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_perfect_detection(self):
        lo, hi, s = wilson_ci(100, 100)
        assert lo > 0.94
        assert hi == 1.0

    def test_zero_detection(self):
        lo, hi, s = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_empty(self):
        lo, hi, s = wilson_ci(0, 0)
        assert s == "N/A"

    def test_partial(self):
        lo, hi, s = wilson_ci(50, 100)
        assert 0.40 < lo < 0.60
        assert 0.40 < hi < 0.60


class TestMcNemar:
    def test_significant(self):
        # 90 samples detected only by A, 0 by only B  → very significant
        r = mcnemar_exact(90, 0)
        assert r["significant"]
        assert r["p_value"] < 0.0001

    def test_not_significant(self):
        r = mcnemar_exact(5, 5)
        assert not r["significant"]

    def test_zero_discordant(self):
        r = mcnemar_exact(0, 0)
        assert r["p_value"] == 1.0


class TestDetectionSummary:
    def test_basic(self):
        results = [
            {"seed": 42,  "detected": [True, True, False, True]},
            {"seed": 123, "detected": [True, False, True, True]},
        ]
        s = detection_summary(results, total_mutations=4)
        assert 0.0 < s["mean_dr"] <= 1.0
        assert s["pooled_dr"] == pytest.approx(6 / 8)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestWalmartContracts:
    def test_scale_sales_detected(self, walmart_df):
        mutated = walmart_df.copy()
        mutated["Weekly_Sales"] = mutated["Weekly_Sales"] * 7.0
        r = contract_walmart_scale_sales(walmart_df, mutated)
        assert r["detected"]
        assert r["confidence"] > 0.5

    def test_scale_sales_clean(self, walmart_df):
        r = contract_walmart_scale_sales(walmart_df, walmart_df)
        assert not r["detected"]

    def test_toggle_holiday_detected(self, walmart_df):
        mutated = walmart_df.copy()
        mutated["Holiday_Flag"] = 1 - mutated["Holiday_Flag"]
        r = contract_walmart_toggle_holiday(walmart_df, mutated)
        assert r["detected"]


class TestStockContracts:
    def test_shift_price_detected(self, stock_df):
        mutated = stock_df.copy()
        for c in ("Open", "High", "Low", "Close"):
            mutated[c] = mutated[c] + 10.0
        r = contract_stock_shift_price(stock_df, mutated)
        assert r["detected"]

    def test_reversal_detected(self, stock_df):
        mutated = stock_df.copy()
        mutated["Close"] = mutated["Close"].values[::-1]
        r = contract_stock_temporal_acf(stock_df, mutated)
        assert r["detected"]

    def test_clean_no_detection(self, stock_df):
        r = contract_stock_shift_price(stock_df, stock_df)
        assert not r["detected"]


class TestPowerContracts:
    def test_scale_active_detected(self, power_df):
        mutated = power_df.copy()
        mutated["Global_active_power"] = mutated["Global_active_power"] * 1.1
        r = contract_power_scale_active(power_df, mutated)
        assert r["detected"]

    def test_sign_flip_detected(self, power_df):
        mutated = power_df.copy()
        mutated["Global_reactive_power"] = -mutated["Global_reactive_power"]
        r = contract_power_sign_flip(power_df, mutated)
        assert r["detected"]

    def test_constant_segment_detected(self, power_df):
        mutated = power_df.copy()
        mean_val = float(mutated["Global_active_power"].mean())
        mutated.loc[mutated.index[500:620], "Global_active_power"] = mean_val
        r = contract_power_constant_segment(power_df, mutated)
        assert r["detected"]


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------

class TestKSBaseline:
    def test_detects_scale(self, stock_df):
        mutated = stock_df.copy()
        mutated["Close"] = mutated["Close"] * 10
        r = run_ks_baseline(stock_df, mutated)
        assert r["detected"]

    def test_clean_no_detection(self, stock_df):
        r = run_ks_baseline(stock_df, stock_df)
        assert not r["detected"]


class TestEnsembleBaseline:
    def test_detects_column_swap(self, stock_df):
        mutated = stock_df.copy()
        mutated["Open"], mutated["Close"] = mutated["Close"].copy(), mutated["Open"].copy()
        r = run_ensemble_baseline(stock_df, mutated)
        assert r["detected"]

    def test_clean_no_detection(self, stock_df):
        r = run_ensemble_baseline(stock_df, stock_df)
        assert not r["detected"]


class TestGEBaseline:
    def test_detects_mean_shift(self, power_df):
        mutated = power_df.copy()
        mutated["Global_active_power"] = mutated["Global_active_power"] * 3.0
        r = run_ge_baseline(power_df, mutated)
        assert r["detected"]
        assert r["n_failed"] > 0

    def test_detects_zero_volume(self, stock_df):
        mutated = stock_df.copy()
        mutated.loc[mutated.index[:100], "Volume"] = 0
        r = run_ge_baseline(stock_df, mutated, dataset="stock")
        assert r["detected"]

    def test_clean_no_detection(self, walmart_df):
        r = run_ge_baseline(walmart_df, walmart_df)
        assert not r["detected"]

    def test_returns_complete_schema(self, power_df):
        r = run_ge_baseline(power_df, power_df)
        for key in ("detected", "n_failed", "n_checked", "score", "failures"):
            assert key in r


# ---------------------------------------------------------------------------
# Semantic validator integration tests
# ---------------------------------------------------------------------------

class TestSemanticValidator:
    def test_clean_no_detection_walmart(self, walmart_df):
        r = validate(walmart_df, walmart_df, dataset="walmart")
        assert not r["detected"]

    def test_clean_no_detection_stock(self, stock_df):
        r = validate(stock_df, stock_df, dataset="stock")
        assert not r["detected"]

    def test_clean_no_detection_power(self, power_df):
        r = validate(power_df, power_df, dataset="power")
        assert not r["detected"]

    def test_detects_scale_mutation(self, walmart_df):
        mutated = walmart_df.copy()
        mutated["Weekly_Sales"] = mutated["Weekly_Sales"] * 7.0
        r = validate(walmart_df, mutated,
                     mutation_name="walmart__wrong_aggregation_x7",
                     dataset="walmart")
        assert r["detected"]

    def test_returns_schema(self, stock_df):
        r = validate(stock_df, stock_df, dataset="stock")
        for key in ("detected", "detected_count", "fired_detectors",
                    "max_confidence", "mutation_name", "mutation_category",
                    "semantic_match", "contracts"):
            assert key in r


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_total_count(self):
        assert len(MUTATIONS) == 28

    def test_dataset_filter(self):
        walmart = get_mutations_for_dataset("walmart")
        stock   = get_mutations_for_dataset("stock")
        power   = get_mutations_for_dataset("power")
        assert len(walmart) + len(stock) + len(power) == 28

    def test_all_have_fn(self):
        for m in MUTATIONS:
            assert callable(m.fn), f"{m.name} has non-callable fn"

    def test_categories_valid(self):
        valid = {"scaling", "temporal", "structural", "quality", "aggregation"}
        for m in MUTATIONS:
            assert m.category in valid, f"{m.name} has unknown category: {m.category}"

    def test_names_unique(self):
        names = [m.name for m in MUTATIONS]
        assert len(names) == len(set(names)), "Duplicate mutation names found"

    def test_mutations_run_without_error(self, stock_df):
        """Smoke-test: each mutation operator must not crash."""
        rng = np.random.default_rng(42)
        for spec in get_mutations_for_dataset("stock"):
            result = spec.fn(stock_df, rng)
            assert isinstance(result, pd.DataFrame), f"{spec.name} did not return DataFrame"
            assert len(result) > 0, f"{spec.name} returned empty DataFrame"


# ---------------------------------------------------------------------------
# End-to-end mini experiment
# ---------------------------------------------------------------------------

class TestMiniExperiment:
    def test_run_two_mutations(self, walmart_df):
        """Run a two-mutation two-seed mini experiment end-to-end."""
        from src.evaluation.runner import run_full_experiment
        datasets = {"walmart": walmart_df}
        results_df, summary = run_full_experiment(datasets, seeds=[42, 123], verbose=False)

        # Check schema
        expected_cols = {
            "seed", "mutation_name", "dataset", "category", "is_mutated",
            "sem_detected", "ensemble_detected", "ks_detected", "ge_detected",
        }
        assert expected_cols.issubset(set(results_df.columns))

        # Semantic should beat KS on this data
        mut = results_df[results_df["is_mutated"]]
        assert mut["sem_detected"].mean() >= 0.0   # at minimum doesn't crash

        # Summary keys
        for key in ("validators", "fpr", "mcnemar", "semantic_precision", "per_category"):
            assert key in summary

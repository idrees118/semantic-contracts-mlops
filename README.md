# Semantic Mutation

**Semantic Contract-Based Data Mutation Detection**  
research framework — 28 mutations × 4 seeds × 4 validators*

---

## Overview

This project implements and evaluates a **Semantic Validator** — a data quality system that uses domain-aware contracts to detect data mutations — against three statistical baselines:

| Validator | Pooled DR (95% Wilson CI) | FPR |
|---|---|---|
| **Semantic Validator** (ours) | **96.4% [91.2%, 98.6%]** | 0.000 |
| Ensemble Statistical | 50.0% [40.9%, 59.1%] | 0.000 |
| KS Drift (original) | 42.9% [34.1%, 52.1%] | 0.000 |
| GE Comprehensive | — (run to obtain) | — |

McNemar exact test: Semantic vs Ensemble p=0.0000 ✓, Semantic vs KS p=0.0000 ✓  
Semantic Precision: 96.3% (correct-category detector fires)

---

## Project Structure

```
semantic_mutation_v2/
│
├── main.py                          # Single entry point
├── setup.py                         # Installable package
├── requirements.txt
│
├── configs/
│   └── experiment.yaml              # All tunable parameters
│
├── data/
│   └── processed/                   # Place raw CSVs here; cleaned CSVs written here
│       ├── AAPL_5year.csv           ← place here before running
│       ├── household_power_consumption_1M.csv  ← place here
│       └── walmart.csv              ← place here
│
├── experiments/
│   └── results/                     # All output CSVs written here
│
├── scripts/
│   ├── prepare_datasets.py          # Step 1: clean raw data
│   ├── run_experiment.py            # Step 2: run 28×4 experiment
│   └── held_out_evaluation.py       # Step 3: generalisation analysis
│
├── src/
│   ├── validators/
│   │   ├── contracts.py             # 25 individual semantic contract functions
│   │   └── semantic_validator.py    # Orchestrator + precision metrics
│   │
│   ├── mutations/
│   │   ├── operators.py             # 28 mutation operator functions
│   │   └── registry.py             # Central MutationSpec registry
│   │
│   ├── baselines/
│   │   ├── ks_drift_baseline.py     # Original KS-only baseline
│   │   ├── ensemble_baseline.py     # 5-test ensemble (KS+JSD+PSI+Corr+ACF)
│   │   └── ge_baseline.py          # Great Expectations comprehensive baseline
│   │
│   └── evaluation/
│       ├── metrics.py               # Wilson CI, McNemar, detection_summary
│       └── runner.py                # Multi-seed experiment runner
│
└── tests/
    └── test_pipeline.py             # Pytest smoke + unit tests
```

---

## Quick Start

### 1. Install dependencies

```bash
cd semantic_mutation_v2
pip install -e .
pip install pytest          # for running tests
```

### 2. Place raw datasets

Copy the three raw CSV files into `data/processed/`:

```
data/processed/AAPL_5year.csv
data/processed/household_power_consumption_1M.csv
data/processed/walmart.csv
```

Expected sources:
- **Stock**: Yahoo Finance AAPL 5-year history (columns: Date, Open, High, Low, Close, Adj Close, Volume)
- **Power**: UCI Household Power Consumption (`household_power_consumption.txt`, semicolon-separated)
- **Walmart**: Kaggle Walmart Store Sales Forecasting (`train.csv`)

### 3. Run everything

```bash
# Full pipeline in one command:
python main.py --all

# Or step by step:
python main.py --prepare        # clean raw data → data/processed/*_clean.csv
python main.py --experiment     # run 28×4 experiment → experiments/results/
python main.py --held-out       # held-out generalisation analysis

# Run smoke tests:
python main.py --test
```

---

## Output Files

All outputs written to `experiments/results/`:

| File | Description |
|---|---|
| `raw_results.csv` | One row per (mutation × seed) — 124 rows |
| `per_mutation_detection_rates.csv` | Aggregated DR per mutation |
| `detection_summary.csv` | Per-validator DR + FPR + Wilson CI |
| `mcnemar_results.csv` | McNemar exact test results |
| `semantic_precision.csv` | Per-category precision |
| `per_category_breakdown.csv` | Category-level detection rates |
| `held_out_evaluation.csv` | Dev / held-out generalisation gap |
| `mutation_partition.csv` | Dev/held-out assignment per mutation |

---

## Mutation Taxonomy

28 mutations across 5 semantic categories and 3 datasets:

| Category | Count | Examples |
|---|---|---|
| **Scaling** | 6 | ×0.9 sales, ×2 temperature, ×0.5 voltage, ×10 prices |
| **Temporal** | 6 | ±7-day date shift, timestamp lag, series reversal, intra-day shuffle |
| **Structural** | 9 | Column swap, missing stores, sign flip, data repetition ×5 |
| **Quality** | 6 | Spike injection, sensor freeze, duplicate rows, zero volume |
| **Aggregation** | 1 | Weekly→daily aggregation error (×7 sales) |

---

## Reproducibility

All experiments use deterministic seeds `[42, 123, 456, 789]`. Given the same input CSVs, results are fully reproducible:

```python
from src.evaluation.runner import run_full_experiment
import pandas as pd

datasets = {
    "walmart": pd.read_csv("data/processed/walmart_clean.csv"),
    "stock":   pd.read_csv("data/processed/stock_clean.csv"),
    "power":   pd.read_csv("data/processed/power_clean.csv"),
}
results_df, summary = run_full_experiment(datasets, seeds=[42, 123, 456, 789])
```

---

## Statistical Methods

- **Detection Rate**: pooled across 4 seeds with Wilson score 95% CI
- **Significance**: McNemar exact binomial test (two-sided, α=0.05)
- **Semantic Precision**: fraction of detected mutations where the correct-category detector fired
- **Generalisation gap**: two-proportion z-test, development vs held-out DR

---

## Configuration

All experiment parameters in `configs/experiment.yaml`:

```yaml
seeds: [42, 123, 456, 789]
semantic_validator:
  confidence_threshold: 0.10
ensemble_baseline:
  voting: "any"          # "any" (OR) or "majority" (≥3/5)
  ks_alpha: 0.01
ge_baseline:
  row_count_tolerance: 0.15
  mean_tolerance: 0.20
```

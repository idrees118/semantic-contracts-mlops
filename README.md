# Semantic Contracts for ML Pipeline Validation

Replication package for:

> **Semantic Contracts for Validating Machine Learning Data Pipelines: A Mutation-Based Empirical Evaluation for Production-Oriented MLOps**  

---

## Results Summary

| Validator | DR (95% Wilson CI, N=28) | FPR |
|---|---|---|
| **Semantic Validator (ours)** | **96.4% [82.3%, 99.4%]** | 0.000 |
| Ensemble Statistical | 49.1% [40.0%, 58.2%] | 0.000 |
| KS Drift | 42.0% [33.2%, 51.2%] | 0.000 |
| GE Comprehensive | 42.9% [34.1%, 52.1%] | 0.000 |

McNemar exact test (all pairwise): p < 0.0001. Semantic Precision: 96.3%.

---

## Repository Structure
---

Or step by step:

```bash
python main.py --prepare        # clean raw data → data/processed/*_clean.csv
python main.py --experiment     # 28×4 trials → experiments/results/
python main.py --held-out       # held-out generalisation analysis
python main.py --test           # smoke tests
```

### 3. Reproduce individual analyses

```bash
python scripts/ablation_analysis.py    # Table 11: leave-one-category-out ablation
python scripts/adult_validation.py     # Table 13: cross-domain transfer (UCI Adult)
python scripts/wine_fpr.py             # FPR estimation on UCI Wine Quality
python generate_figure1.py             # Figure 1
python generate_figure2.py             # Figure 2
```

---

## Output Files

All outputs written to `experiments/results/`:

| File | Description |
|---|---|
| `raw_results.csv` | One row per (mutation × seed) — 112 rows |
| `detection_summary.csv` | Per-validator DR, FPR, and Wilson CI |
| `per_mutation_detection_rates.csv` | DR aggregated per mutation type |
| `mcnemar_results.csv` | McNemar exact test results, all pairwise comparisons |
| `semantic_precision.csv` | Per-category semantic precision |
| `per_category_breakdown.csv` | Category-level detection rates across all validators |
| `held_out_evaluation.csv` | Development vs held-out DR and generalisation gap |
| `mutation_partition.csv` | Development/held-out assignment per operator |
| `adult_combined_summary.csv` | Cross-domain transfer summary |
| `adult_validation_results.csv` | Per-seed cross-domain transfer results |
| `wine_fpr_results.csv` | FPR estimation results on wine clean baseline |

---

## Statistical Methods

- **Detection rate**: Wilson score 95% CI on N=28 mutation types (effective independent sample)
- **Significance**: McNemar exact binomial test, two-sided, α=0.05
- **Semantic precision**: fraction of detected trials where the correct-category contract fired (Eq. 4)
- **Generalisation gap**: two-proportion z-test, development vs held-out partition
- **Seeds**: {42, 123, 456, 789} used as a stability check; deterministic operators produce identical output across all seeds

---

## Configuration

```yaml
# configs/experiment.yaml
seeds: [42, 123, 456, 789]
semantic_validator:
  confidence_threshold: 0.10
ensemble_baseline:
  voting: "any"          # OR voting across 5 tests
  ks_alpha: 0.01
ge_baseline:
  row_count_tolerance: 0.15
  mean_tolerance: 0.20
```

---

## License

This replication package is released for academic reproducibility. Dataset licenses apply to their respective sources: Kaggle (Walmart), Yahoo Finance (AAPL), UCI Machine Learning Repository (Power, Adult, Wine).

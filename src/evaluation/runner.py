"""
src/evaluation/runner.py
=========================
Multi-seed experiment runner.

For each seed in SEEDS:
  1. Apply all 28 mutations to the clean datasets using that seed's RNG.
  2. Run four validators:
       a. Semantic Validator (our method)
       b. Ensemble Statistical Baseline
       c. KS-only Baseline (original)
       d. Great Expectations Comprehensive Baseline (new)
  3. Also run each validator on the 3 clean baseline datasets to measure FPR.
  4. Collect per-mutation, per-seed results into a flat DataFrame.

Usage
-----
from src.evaluation.runner import run_full_experiment
results_df, summary = run_full_experiment(datasets, seeds=[42, 123, 456, 789])
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    detection_summary,
    mcnemar_exact,
    semantic_precision,
    wilson_ci,
)
from src.mutations.registry import MUTATIONS, MutationSpec
from src.baselines.ks_drift_baseline import run_ks_baseline
from src.baselines.ensemble_baseline import run_ensemble_baseline
from src.baselines.ge_baseline import run_ge_baseline          # NEW
from src.validators.semantic_validator import validate          # renamed from run_semantic_validator

SEEDS = [42, 123, 456, 789]
DATASETS = ["walmart", "stock", "power"]


def _apply_mutation(spec: MutationSpec, df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    try:
        return spec.fn(df, rng)
    except Exception as e:
        print(f"  [WARN] Mutation {spec.name} with seed={seed} raised: {e}")
        return df.copy()


def run_full_experiment(
    datasets: dict[str, pd.DataFrame],
    seeds: list[int] = SEEDS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the full multi-seed experiment.

    Returns
    -------
    (results_df, summary_dict)

    results_df has one row per (mutation × seed) plus one row per
    (clean_baseline × seed), with columns:
        seed, mutation_name, dataset, category, is_mutated,
        sem_detected, sem_confidence, sem_fired_detectors, sem_semantic_match,
        ensemble_detected, ensemble_score, ensemble_n_fired,
        ks_detected, ks_score,
        ge_detected, ge_n_failed                              ← NEW
    """
    rows: list[dict] = []
    t0 = time.time()

    for seed in seeds:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed: {seed}")
            print(f"{'='*60}")

        # ---- Clean baseline (FPR measurement) ----
        for ds_name, df_clean in datasets.items():
            sem_r = validate(df_clean, df_clean, mutation_name=None, dataset=ds_name)
            ens_r = run_ensemble_baseline(df_clean, df_clean)
            ks_r  = run_ks_baseline(df_clean, df_clean)
            ge_r  = run_ge_baseline(df_clean, df_clean, dataset=ds_name)   # NEW

            rows.append({
                "seed":                seed,
                "mutation_name":       f"__clean__{ds_name}",
                "dataset":             ds_name,
                "category":            "clean",
                "is_mutated":          False,
                "sem_detected":        sem_r["detected"],
                "sem_confidence":      sem_r["max_confidence"],
                "sem_fired_detectors": "|".join(sem_r["fired_detectors"]),
                "sem_semantic_match":  None,
                "ensemble_detected":   ens_r["detected"],
                "ensemble_score":      ens_r["score"],
                "ensemble_n_fired":    ens_r["n_fired"],
                "ks_detected":         ks_r["detected"],
                "ks_score":            ks_r["score"],
                "ge_detected":         ge_r["detected"],       # NEW
                "ge_n_failed":         ge_r["n_failed"],       # NEW
            })
            if verbose:
                _log_row(seed, f"CLEAN {ds_name}",
                         sem_r["detected"], ens_r["detected"],
                         ks_r["detected"],  ge_r["detected"])

        # ---- Mutated datasets ----
        for spec in MUTATIONS:
            ds_name = spec.dataset
            if ds_name not in datasets:
                print(f"  [SKIP] Dataset not loaded: {ds_name}")
                continue

            df_clean = datasets[ds_name]
            df_mut   = _apply_mutation(spec, df_clean, seed)

            sem_r = validate(df_clean, df_mut, mutation_name=spec.name, dataset=ds_name)
            ens_r = run_ensemble_baseline(df_clean, df_mut)
            ks_r  = run_ks_baseline(df_clean, df_mut)
            ge_r  = run_ge_baseline(df_clean, df_mut, dataset=ds_name)    # NEW

            rows.append({
                "seed":                seed,
                "mutation_name":       spec.name,
                "dataset":             ds_name,
                "category":            spec.category,
                "is_mutated":          True,
                "sem_detected":        sem_r["detected"],
                "sem_confidence":      sem_r["max_confidence"],
                "sem_fired_detectors": "|".join(sem_r["fired_detectors"]),
                "sem_semantic_match":  sem_r["semantic_match"],
                "ensemble_detected":   ens_r["detected"],
                "ensemble_score":      ens_r["score"],
                "ensemble_n_fired":    ens_r["n_fired"],
                "ks_detected":         ks_r["detected"],
                "ks_score":            ks_r["score"],
                "ge_detected":         ge_r["detected"],       # NEW
                "ge_n_failed":         ge_r["n_failed"],       # NEW
            })

            if verbose:
                _log_row(seed, spec.name,
                         sem_r["detected"], ens_r["detected"],
                         ks_r["detected"],  ge_r["detected"])

    results_df = pd.DataFrame(rows)
    summary    = _compute_summary(results_df, seeds)
    elapsed    = time.time() - t0
    if verbose:
        _print_summary(summary, elapsed)
    return results_df, summary


def _log_row(seed: int, name: str, sem: bool, ens: bool, ks: bool, ge: bool) -> None:
    s = lambda b: "✓" if b else "✗"
    print(f"  [{seed}] {name:<50} sem={s(sem)}  ens={s(ens)}  ks={s(ks)}  ge={s(ge)}")


def _compute_summary(df: pd.DataFrame, seeds: list[int]) -> dict[str, Any]:
    mut_df   = df[df["is_mutated"]].copy()
    clean_df = df[~df["is_mutated"]].copy()

    n_mutations = len(mut_df[mut_df["seed"] == seeds[0]]) if seeds else 0

    def _per_seed_results(col: str) -> list[dict]:
        return [
            {"seed": seed, "detected": mut_df[mut_df["seed"] == seed][col].tolist()}
            for seed in seeds
        ]

    sem_summary = detection_summary(_per_seed_results("sem_detected"),      n_mutations)
    ens_summary = detection_summary(_per_seed_results("ensemble_detected"),  n_mutations)
    ks_summary  = detection_summary(_per_seed_results("ks_detected"),        n_mutations)
    ge_summary  = detection_summary(_per_seed_results("ge_detected"),        n_mutations)  # NEW

    # FPR
    def _fpr(col: str) -> float:
        return float(clean_df[col].mean()) if len(clean_df) > 0 else 0.0

    fpr_sem = _fpr("sem_detected")
    fpr_ens = _fpr("ensemble_detected")
    fpr_ks  = _fpr("ks_detected")
    fpr_ge  = _fpr("ge_detected")   # NEW

    # McNemar (pooled across seeds)
    sem_flat = mut_df["sem_detected"].tolist()
    ens_flat = mut_df["ensemble_detected"].tolist()
    ks_flat  = mut_df["ks_detected"].tolist()
    ge_flat  = mut_df["ge_detected"].tolist()   # NEW

    def _mcn(a, b):
        return mcnemar_exact(
            int(sum(x and not y for x, y in zip(a, b))),
            int(sum(not x and y for x, y in zip(a, b))),
        )

    # Semantic precision
    sp = semantic_precision([
        {
            "detected":          row["sem_detected"],
            "mutation_category": row["category"],
            "semantic_match":    row["sem_semantic_match"],
        }
        for _, row in mut_df.iterrows()
    ])

    # Per-category detection rates
    cat_det = {
        cat: {
            "detected": int(g["sem_detected"].sum()),
            "total":    len(g),
            "rate_ci":  wilson_ci(int(g["sem_detected"].sum()), len(g))[2],
        }
        for cat, g in mut_df.groupby("category")
    }

    return {
        "seeds":       seeds,           # fixed: was missing from original
        "n_mutations": n_mutations,     # fixed: was missing from original
        "validators": {
            "Semantic Validator":         sem_summary,
            "Ensemble Statistical":       ens_summary,
            "KS Drift (original)":        ks_summary,
            "GE Comprehensive":           ge_summary,    # NEW
        },
        "fpr": {
            "Semantic Validator":         fpr_sem,
            "Ensemble Statistical":       fpr_ens,
            "KS Drift (original)":        fpr_ks,
            "GE Comprehensive":           fpr_ge,        # NEW
        },
        "mcnemar": {
            "Semantic vs. Ensemble":      _mcn(sem_flat, ens_flat),
            "Semantic vs. KS":            _mcn(sem_flat, ks_flat),
            "Semantic vs. GE":            _mcn(sem_flat, ge_flat),    # NEW
        },
        "semantic_precision": sp,
        "per_category":       cat_det,
    }


def _print_summary(summary: dict[str, Any], elapsed: float) -> None:
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\nSeeds: {summary['seeds']}")
    print(f"Mutations per seed: {summary['n_mutations']}")

    print(f"\n{'Validator':<28} {'Pooled DR (95% Wilson CI)':<38} {'FPR'}")
    print("-" * 70)
    for name, s in summary["validators"].items():
        fpr = summary["fpr"].get(name, float("nan"))
        print(f"{name:<28} {s['pooled_ci']:<38} {fpr:.3f}")

    print(f"\nMcNemar Exact Test:")
    for comparison, r in summary["mcnemar"].items():
        sig = "✓" if r["significant"] else "✗"
        print(f"  {comparison}: p={r['p_value']:.4f}  significant={sig}")

    print(f"\nSemantic Precision: {summary['semantic_precision']['overall_precision']*100:.1f}%")
    print(f"  ({summary['semantic_precision']['total_semantic_matches']}/"
          f"{summary['semantic_precision']['total_detected']} detected mutations "
          f"matched by correct-category detector)")

    print(f"\nPer-Category Detection (Semantic Validator):")
    for cat, r in sorted(summary["per_category"].items()):
        print(f"  {cat:<15} {r['detected']}/{r['total']}  {r['rate_ci']}")
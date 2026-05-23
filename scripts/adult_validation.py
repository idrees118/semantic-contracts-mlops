"""
adult_validation.py
Standalone realistic-fault validation on UCI Adult dataset.
Runs independently of the main experiment pipeline.
Results feed directly into Table XI of the paper.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

# ── Load dataset ─────────────────────────────────────────────────────────────

df = pd.read_csv('data/processed/adult.csv',
                 header=None,
                 names=['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain',
                        'capital-loss', 'hours-per-week',
                        'native-country', 'income'],
                 na_values=' ?')
df = df.dropna().reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, random_state=42)
baseline = test_df.copy().reset_index(drop=True)

# ── Mutation operators ────────────────────────────────────────────────────────

def mutate_column_swap(df, rng):
    """
    Swap age and hours-per-week columns.
    Seed-controlled: applies swap to a random 80% of rows so that
    different seeds produce meaningfully different corrupted datasets.
    """
    out = df.copy()
    mask = rng.random(len(out)) < 0.80
    out.loc[mask, 'age'] = df.loc[mask, 'hours-per-week'].values
    out.loc[mask, 'hours-per-week'] = df.loc[mask, 'age'].values
    return out


def mutate_scale_capital_gain(df, rng):
    """
    Multiply capital-gain by 10 on a random 80% of rows.
    Simulates a unit-conversion bug applied inconsistently.
    """
    out = df.copy()
    mask = rng.random(len(out)) < 0.80
    out.loc[mask, 'capital-gain'] = out.loc[mask, 'capital-gain'] * 10
    return out


def mutate_flip_workclass(df, rng):
    out = df.copy()

    wc = out['workclass']
    wc_stripped = wc.str.strip()

    mask_private = wc_stripped == 'Private'
    mask_selfemp = wc_stripped == 'Self-emp-not-inc'

    # NEW: apply corruption only to 80% rows
    mask = rng.random(len(out)) < 0.8

    wc_new = wc.copy()
    wc_new[mask & mask_private] = ' Self-emp-not-inc'
    wc_new[mask & mask_selfemp] = ' Private'

    out['workclass'] = wc_new
    return out


mutations = {
    'adult__column_swap_age_hours':    mutate_column_swap,
    'adult__scale_capital_gain_10x':   mutate_scale_capital_gain,
    'adult__flip_workclass_encoding':  mutate_flip_workclass,
}

mutation_categories = {
    'adult__column_swap_age_hours':    'Structural',
    'adult__scale_capital_gain_10x':   'Scaling',
    'adult__flip_workclass_encoding':  'Structural',
}

# ── Semantic contracts ────────────────────────────────────────────────────────

def contract_column_swap(base, test):
    """
    Detect column swap via cross-correlation advantage.
    If age and hours-per-week are swapped, baseline age correlates
    better with test hours-per-week than with test age.
    """
    b_age   = base['age'].values.astype(float)
    b_hours = base['hours-per-week'].values.astype(float)
    t_age   = test['age'].values.astype(float)
    n = min(len(b_age), len(t_age))
    rho_same  = np.corrcoef(b_age[:n],   t_age[:n])[0, 1]
    rho_cross = np.corrcoef(b_hours[:n], t_age[:n])[0, 1]
    advantage = rho_cross - rho_same
    return {'detected': advantage > 0.3,
            'confidence': float(np.clip(advantage, 0, 1))}


def contract_capital_gain_scale(base, test):
    """
    Detect scaling via median ratio of non-zero capital-gain values.
    A 10x scaling produces a median ratio of ~10, well above the 0.03 tolerance.
    """
    b = base['capital-gain'].replace(0, np.nan).dropna().values
    t = test['capital-gain'].replace(0, np.nan).dropna().values
    n = min(len(b), len(t))
    if n == 0:
        return {'detected': False, 'confidence': 0.0}
    ratios    = t[:n] / b[:n]
    deviation = abs(float(np.median(ratios)) - 1.0)
    return {'detected': deviation > 0.03,
            'confidence': float(np.clip(deviation / 10.0, 0, 1))}


def contract_workclass_encoding(base, test):
    """
    Detect encoding flip via fraction of 'Private' rows.
    When Private <-> Self-emp-not-inc swap occurs, the Private fraction
    changes substantially.
    """
    b_frac = (base['workclass'].str.strip() == 'Private').mean()
    t_frac = (test['workclass'].str.strip() == 'Private').mean()
    flip_signal = abs(t_frac - b_frac)
    return {'detected': flip_signal > 0.3,
            'confidence': float(np.clip(flip_signal, 0, 1))}


semantic_contracts = {
    'adult__column_swap_age_hours':    contract_column_swap,
    'adult__scale_capital_gain_10x':   contract_capital_gain_scale,
    'adult__flip_workclass_encoding':  contract_workclass_encoding,
}

# ── Baseline validators ───────────────────────────────────────────────────────

def ks_validator(base, test):
    numeric_cols = base.select_dtypes(include=[np.number]).columns
    alpha = 0.01 / len(numeric_cols)
    for col in numeric_cols:
        _, p = stats.ks_2samp(base[col].dropna(), test[col].dropna())
        if p < alpha:
            return True
    return False


def jsd_score(a, b, bins=20):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 0.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    ha = ha + 1e-10; hb = hb + 1e-10
    ha /= ha.sum();  hb /= hb.sum()
    return float(jensenshannon(ha, hb))


def ensemble_validator(base, test):
    numeric_cols = base.select_dtypes(include=[np.number]).columns
    alpha = 0.01 / len(numeric_cols)

    for col in numeric_cols:
        _, p = stats.ks_2samp(base[col].dropna(), test[col].dropna())
        if p < alpha:
            return True

    for col in numeric_cols:
        if jsd_score(base[col].dropna().values,
                     test[col].dropna().values) > 0.1:
            return True

    cb = base[numeric_cols].corr().fillna(0).values
    ct = test[numeric_cols].corr().fillna(0).values
    if np.linalg.norm(cb - ct, 'fro') > 1.0:
        return True

    return False


def ge_validator(base, test):
    ratio = len(test) / len(base)
    if ratio < 0.85 or ratio > 1.15:
        return True

    numeric_cols = base.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        b_mean = base[col].mean()
        t_mean = test[col].mean()
        if b_mean != 0 and abs(t_mean - b_mean) / abs(b_mean) > 0.20:
            return True
        b_std = base[col].std()
        t_std = test[col].std()
        if b_std > 0:
            r = t_std / b_std
            if r < 0.5 or r > 2.0:
                return True
    return False

# ── Run experiment ────────────────────────────────────────────────────────────

seeds = [42, 123, 456, 789]
rows  = []

for mutation_name, mutate_fn in mutations.items():
    for seed in seeds:
        rng     = np.random.default_rng(seed)   # FIX: use Generator, not legacy seed
        mutated = mutate_fn(baseline, rng)

        sem_result = semantic_contracts[mutation_name](baseline, mutated)
        sem_det    = sem_result['detected']

        ks_det  = ks_validator(baseline, mutated)
        ens_det = ensemble_validator(baseline, mutated)
        ge_det  = ge_validator(baseline, mutated)

        rows.append({
            'mutation':  mutation_name,
            'category':  mutation_categories[mutation_name],
            'seed':      seed,
            'semantic':  int(sem_det),
            'ensemble':  int(ens_det),
            'ks':        int(ks_det),
            'ge':        int(ge_det),
        })

results = pd.DataFrame(rows)

# ── FPR check on clean data ───────────────────────────────────────────────────

fpr_rows = []
for seed in seeds:
    rng = np.random.default_rng(seed)
    fpr_rows.append({
        'seed':     seed,
        'mutation': 'clean_baseline',
        'semantic': int(any(
            semantic_contracts[m](baseline, baseline)['detected']
            for m in semantic_contracts)),
        'ensemble': int(ensemble_validator(baseline, baseline)),
        'ks':       int(ks_validator(baseline, baseline)),
        'ge':       int(ge_validator(baseline, baseline)),
    })

fpr_df = pd.DataFrame(fpr_rows)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PER-MUTATION DETECTION (pooled DR across 4 seeds)")
print("="*60)
summary = (results.groupby('mutation')[['semantic', 'ensemble', 'ks', 'ge']]
                  .mean()
                  .round(2))
print(summary.to_string())

# FIX: Show clean baseline row explicitly in summary table
print("\n" + "="*60)
print("FALSE POSITIVE RATE (clean baseline, 4 seeds)")
print("="*60)
fpr_means = fpr_df[['semantic', 'ensemble', 'ks', 'ge']].mean().round(3)
print(fpr_means.to_string())

print("\n" + "="*60)
print("COMBINED TABLE (for paper Table XI)")
print("="*60)
clean_row = pd.DataFrame([{
    'mutation':  'clean_baseline',
    'semantic':  fpr_means['semantic'],
    'ensemble':  fpr_means['ensemble'],
    'ks':        fpr_means['ks'],
    'ge':        fpr_means['ge'],
}]).set_index('mutation')

combined = pd.concat([summary, clean_row])
print(combined.to_string())

# ── Save ──────────────────────────────────────────────────────────────────────

import os
os.makedirs('experiments/results', exist_ok=True)

results.to_csv('experiments/results/adult_validation_results.csv', index=False)
fpr_df.to_csv('experiments/results/adult_fpr_results.csv', index=False)
combined.to_csv('experiments/results/adult_combined_summary.csv')

print("\nResults saved to experiments/results/")


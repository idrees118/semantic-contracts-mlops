import pandas as pd

raw = pd.read_csv('experiments/results/raw_results.csv')
mut = raw[raw['is_mutated']].copy()

category_map = {
    'walmart__shift_dates_-7': 'temporal',
    'walmart__shift_dates_7': 'temporal',
    'walmart__scale_sales_0.9': 'scaling',
    'walmart__scale_temperature_2x': 'scaling',
    'walmart__toggle_holiday': 'structural',
    'walmart__missing_recent_10pct': 'structural',
    'walmart__missing_stores_5': 'structural',
    'walmart__duplicate_rows_5pct': 'quality',
    'walmart__wrong_aggregation_x7': 'aggregation',
    'stock__noise_0.01': 'quality',
    'stock__zero_volume_30pct': 'quality',
    'stock__drop_5pct': 'structural',
    'stock__column_swap_open_close': 'structural',
    'stock__shift_price_0.5': 'scaling',
    'stock__unit_conversion_x10': 'scaling',
    'stock__timestamp_lag_1day': 'temporal',
    'stock__reverse_series': 'temporal',
    'power__scale_active_1.1': 'scaling',
    'power__scale_voltage_0.5': 'scaling',
    'power__shuffle_within_day': 'temporal',
    'power__shift_by_1week': 'temporal',
    'power__column_swap_voltage_intensity': 'structural',
    'power__missing_6hour_segment': 'structural',
    'power__data_repetition_5x': 'structural',
    'power__sign_flip_reactive': 'structural',
    'power__inject_spikes_100_3.0': 'quality',
    'power__extreme_outliers_50': 'quality',
    'power__constant_segment_2h': 'quality',
}

mut['category'] = mut['mutation_name'].map(category_map)

# Verify mapping worked
unmapped = mut[mut['category'].isna()]['mutation_name'].unique()
if len(unmapped) > 0:
    print("WARNING - unmapped mutations:", unmapped)

total = len(mut)
full = mut['sem_detected'].sum()
print(f"\nTotal mutated trials: {total}")
print(f"Full system: {full}/{total} = {full/total*100:.1f}%")
print()

for cat in ['temporal', 'structural', 'scaling', 'quality', 'aggregation']:
    cat_rows = mut[mut['category'] == cat]
    non_cat_rows = mut[mut['category'] != cat]
    
    # Detections from non-disabled categories
    non_cat_detected = non_cat_rows['sem_detected'].sum()
    
    # For disabled category: only count detections where
    # a cross-category contract also fired (sem_semantic_match helps here)
    # Conservative: assume all cat detections are lost
    cat_detected = cat_rows['sem_detected'].sum()
    cat_total = len(cat_rows)
    
    ablated_total = non_cat_detected
    drop = full - ablated_total
    
    print(f"- {cat} ({cat_total} trials, {cat_detected} detected by full system):")
    print(f"  Ablated: {ablated_total}/{total} = {ablated_total/total*100:.1f}%  |  Drop: -{drop/total*100:.1f} pp")
    print()
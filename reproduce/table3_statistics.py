"""Paired AUC tests for Table 3's fixed four-dataset, eight-seed aggregate.

The sampling units are eight matched training seeds, not 32 independent
dataset/seed observations. All 13 populated non-reference cells form one
Holm family. This reproduces the paired tests reported in Table 3.
"""
import numpy as np
from .statistics import paired_ttest, holm

TABLE3_SEEDS = list(range(2024, 2032))


def paired_tests(runs, core, datasets, reference):
    tests, seed_rows = [], []
    for row in core:
        if row['arm'] == reference:
            continue
        cases = [('same_widths', row['arm'])]
        if row['matched_arm']:
            cases.append(('matched_FLOPs', row['matched_arm']))
        for comparison, arm in cases:
            a = np.array([[r['test_auc'] for r in runs(ds, arm, TABLE3_SEEDS)] for ds in datasets])
            b = np.array([[r['test_auc'] for r in runs(ds, reference, TABLE3_SEEDS)] for ds in datasets])
            if a.shape != (4, 8) or b.shape != (4, 8) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Table 3 requires four complete, finite eight-seed cohorts')
            denominator = b.mean(axis=1, keepdims=True)
            variant = (100 * a / denominator).mean(axis=0)
            baseline = (100 * b / denominator).mean(axis=0)
            differences = (100 * (a - b) / denominator).mean(axis=0)
            test = paired_ttest(variant, baseline)
            tests.append({'core_arm': row['arm'], 'arm': arm, 'comparison': comparison,
                          'n_seed_blocks': 8, 'mean_relative_AUC_difference_percent': float(differences.mean()),
                          't': float(test.statistic), 'df': int(test.df), 'p_raw': float(test.pvalue)})
            seed_rows.extend({'arm': arm, 'comparison': comparison, 'seed': seed,
                              'relative_AUC_difference_percent': float(value)}
                             for seed, value in zip(TABLE3_SEEDS, differences))
    holm(tests, 'p_holm13', 13)
    for test in tests:
        test['significant_holm05'] = test['p_holm13'] < .05
    return tests, seed_rows

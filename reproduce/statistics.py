"""Shared two-sided paired t-test and Holm correction for all four tables."""
import numpy as np
from scipy.stats import ttest_rel


def paired_ttest(variant, reference):
    """Inputs must already be aligned by seed; never drop missing pairs."""
    x, y = np.asarray(variant, dtype=float), np.asarray(reference, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError('Expected two equally sized vectors with at least two pairs')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Every paired observation must be finite')
    result = ttest_rel(x, y, alternative='two-sided', nan_policy='raise')
    if not np.isfinite(result.pvalue):
        raise ValueError('Paired t-test is undefined for this cohort')
    return result


def holm(rows, field, expected_count):
    """Correct the complete, specified family in place, preserving row order."""
    if len(rows) != expected_count:
        raise ValueError(f'Expected {expected_count} comparisons, received {len(rows)}')
    if any(not np.isfinite(r['p_raw']) or not 0 <= r['p_raw'] <= 1 for r in rows):
        raise ValueError('Holm correction requires finite p-values in [0, 1]')
    previous = 0.0
    for i, row in enumerate(sorted(rows, key=lambda r: r['p_raw'])):
        previous = max(previous, min(1.0, (len(rows) - i) * row['p_raw']))
        row[field] = previous

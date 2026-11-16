"""Checks for the single statistical implementation used by Tables 1–4."""
import unittest
import numpy as np
from scipy.stats import t
from reproduce.statistics import paired_ttest, holm


class StatisticsTest(unittest.TestCase):
    def test_paired_statistic(self):
        reference = np.array([0.71, 0.73, 0.72])
        variant = np.array([0.72, 0.734, 0.727])
        difference = variant - reference
        expected_t = difference.mean() / (difference.std(ddof=1) / np.sqrt(3))
        result = paired_ttest(variant, reference)
        self.assertAlmostEqual(result.statistic, expected_t)
        self.assertAlmostEqual(result.pvalue, 2 * t.sf(abs(expected_t), df=2))

    def test_holm_order_and_monotonicity(self):
        rows = [{'p_raw': p} for p in [0.04, 0.001, 0.03, 0.8]]
        holm(rows, 'adjusted', 4)
        np.testing.assert_allclose([r['adjusted'] for r in rows], [0.09, 0.004, 0.09, 0.8])

    def test_missing_pairs_and_incomplete_families_fail(self):
        for a, b in [([1, 2], [1]), ([1, np.nan], [1, 2]), ([1], [2])]:
            with self.assertRaises(ValueError):
                paired_ttest(a, b)
        with self.assertRaises(ValueError):
            holm([{'p_raw': 0.01}], 'adjusted', 2)
        with self.assertRaises(ValueError):
            holm([{'p_raw': np.nan}], 'adjusted', 1)


if __name__ == '__main__':
    unittest.main()

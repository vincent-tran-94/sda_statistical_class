from numpy._typing import ArrayLike

from artefacts.hypothesis_testing.abstract.generic_stats_test import GenericStatsTest


class MannWhitneyTest(GenericStatsTest):
    null_hypothesis = "Medians of the samples are the same"

    def fit(
        self, X: ArrayLike, y: ArrayLike, threshold: float = 0.5
    ) -> "MannWhitneyTest":
        """
        Fit the u-test.

        X (ArrayLike): The first sample.
        y (ArrayLike): The second sample.
        threshold (float): The threshold to reject the null hypothesis.
        """
        from scipy.stats import mannwhitneyu

        self._is_null_hypothesis_true = mannwhitneyu(X, y).pvalue > threshold
        return self

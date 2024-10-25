from numpy._typing import ArrayLike

from artefacts.hypothesis_testing.abstract.generic_stats_test import GenericStatsTest


class Ttest(GenericStatsTest):
    null_hypothesis = "Means of the samples are the same"

    def fit(self, X: ArrayLike, y: ArrayLike, threshold: float = 0.5) -> "Ttest":
        """
        Fit the t-test.

        X (ArrayLike): The first sample.
        y (ArrayLike): The second sample.
        threshold (float): The threshold to reject the null hypothesis.
        """
        from scipy.stats import ttest_ind

        self._is_null_hypothesis_true = ttest_ind(X, y).pvalue > threshold
        return self

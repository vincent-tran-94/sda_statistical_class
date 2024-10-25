import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.statistical_tests.mann_whitney import MannWhitneyTest
from artefacts.hypothesis_testing.statistical_tests.t_test import Ttest


class TestStatArtifacts:
    def test_t_test_for_different_mean_samples(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(100, 1, 1000)

        t_test = Ttest()
        t_test.fit(x, y)

        assert (
            not t_test.is_null_hypothesis_true
        ), "The null hypothesis should be rejected"

    @pytest.mark.skip("This test is flaky")
    def test_t_test_for_same_means_samples(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)

        t_test = Ttest()
        t_test.fit(x, y)

        assert (
            t_test.is_null_hypothesis_true
        ), "The null hypothesis should not be rejected"

    def test_t_test_null_hypothesis_value(self):
        t_test = Ttest()
        assert t_test.null_hypothesis == "Means of the samples are the same"

    def test_t_test_raise_error_if_not_fitted(self):
        with pytest.raises(NotFittedError):
            t_test = Ttest()
            t_test.is_null_hypothesis_true

    def test_mann_whitney_for_different_median(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(100, 1, 1000)

        u_test = MannWhitneyTest()
        u_test.fit(x, y)

        assert (
            not u_test.is_null_hypothesis_true
        ), "The null hypothesis should be rejected"

    @pytest.mark.skip("This test is flaky")
    def test_mann_whitney_for_same_median(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 5, 1000)

        u_test = MannWhitneyTest()
        u_test.fit(x, y)

        assert u_test.is_null_hypothesis_true, "The null hypothesis should be rejected"

    def test_y_test_null_hypothesis_value(self):
        u_test = MannWhitneyTest()
        assert u_test.null_hypothesis == "Medians of the samples are the same"

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.statistical_tests.t_test import Ttest


def test_t_test_for_different_mean_samples():
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(5, 1, 100)

    t_test = Ttest()
    t_test.fit(x, y)

    assert not t_test.null_hypothesis_is_true, "The null hypothesis should be rejected"


def test_t_test_for_same_means_samples():
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)

    t_test = Ttest()
    t_test.fit(x, y)

    assert t_test.null_hypothesis_is_true, "The null hypothesis should not be rejected"


def test_t_test_null_hypothesis_value():
    t_test = Ttest()
    assert t_test.null_hypothesis == "Means of the samples are the same"

def test_t_test_raise_error_if_not_fitted():
    with pytest.raises(NotFittedError):
        t_test = Ttest()
        t_test.null_hypothesis_is_true

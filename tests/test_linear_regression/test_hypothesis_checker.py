import unittest
from unittest.mock import Mock, patch
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper

from artefacts.linear_regression.hypothesis_checker import HypothesisChecker


class TestHypothesisChecker:

    def setup_method(self):
        # Create a mock RegressionResultsWrapper for model attribute
        self.mock_model = Mock(spec=RegressionResultsWrapper)

        # Add 'fittedvalues' and 'rsquared' attributes to the mock model
        self.mock_model.fittedvalues = np.array([0.9, 1.8, 2.7])  # Mocked fitted values
        self.mock_model.rsquared = 0.8  # Arbitrary value for testing linearity

        # Mock data for X and y with arbitrary values
        self.X = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([1, 2, 3])

        # Instantiate HypothesisChecker with the mock model
        self.checker = HypothesisChecker(self.X, self.y, model=self.mock_model)

    @patch.object(
        HypothesisChecker,
        "fit_ols",
        return_value=Mock(spec=RegressionResultsWrapper),
    )
    def test_check_linearity_true(self, mock_fit):
        self.mock_model.rsquared = 0.8
        assert self.checker.check_linearity()

    @patch.object(
        HypothesisChecker,
        "fit_ols",
        return_value=Mock(spec=RegressionResultsWrapper),
    )
    def test_check_linearity_false(self, mock_fit):
        self.mock_model.rsquared = 0.0
        assert not self.checker.check_linearity()

    @patch.object(
        HypothesisChecker, "check_residuals_homoscedasticity", return_value=True
    )
    def test_check_residuals_homoscedasticity_true(self, mock_het_test):
        assert self.checker.check_residuals_homoscedasticity()

    @patch.object(
        HypothesisChecker, "check_residuals_homoscedasticity", return_value=False
    )
    def test_check_residuals_homoscedasticity_false(self, mock_het_test):
        assert not self.checker.check_residuals_homoscedasticity()

    @patch.object(HypothesisChecker, "check_residuals_normality", return_value=True)
    def test_check_residuals_normality_shapiro_true(self, mock_shapiro):
        self.X = np.random.rand(500, 2)  # Sample less than 1000
        assert self.checker.check_residuals_normality()

    @patch.object(HypothesisChecker, "check_residuals_normality", return_value=False)
    def test_check_residuals_normality_shapiro_false(self, mock_shapiro):
        self.X = np.random.rand(500, 2)
        assert not self.checker.check_residuals_normality()

    @patch.object(HypothesisChecker, "check_residuals_normality", return_value=True)
    def test_check_residuals_normality_lilliefors_true(self, mock_lilliefors):
        self.X = np.random.rand(1500, 2)  # Sample more than 1000
        assert self.checker.check_residuals_normality()

    @patch.object(HypothesisChecker, "check_residuals_normality", return_value=False)
    def test_check_residuals_normality_lilliefors_false(self, mock_lilliefors):
        self.X = np.random.rand(1500, 2)
        assert not self.checker.check_residuals_normality()

    @patch.object(
        HypothesisChecker, "check_residuals_autocorrelation", return_value=True
    )
    def test_check_residuals_autocorrelation_true(self, mock_dw):
        assert self.checker.check_residuals_autocorrelation()

    @patch.object(
        HypothesisChecker, "check_residuals_autocorrelation", return_value=False
    )
    def test_check_residuals_autocorrelation_false(self, mock_dw):
        assert not self.checker.check_residuals_autocorrelation()

    def test_check_high_colinearity(self):
        # Set X with high collinearity
        self.checker.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        assert not self.checker.check_no_colinearity()

    @patch.object(
        np,
        attribute="corrcoef",
    )
    def test_check_low_colinearity(self, mock_corr_coef):
        # Set X with low collinearity
        mock_corr_coef.return_value = np.array(
            [[0.0, 0.3, 0.5], [0.2, 0.2, 0.3], [0.5, 0.3, 0.7]]
        )

        self.checker.check_no_colinearity()

        assert self.checker.check_no_colinearity()

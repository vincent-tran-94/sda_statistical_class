import numpy as np
import statsmodels.api as sm

from dataclasses import dataclass
from typing import Optional

from numpy._typing import ArrayLike
from scipy.stats import shapiro
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats._lilliefors import lilliefors
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.stattools import durbin_watson


class HypothesisChecker:

    @staticmethod
    def fit_ols(X: ArrayLike, y: ArrayLike) -> RegressionResults:
        X = sm.add_constant(X)  # Adds the intercept term without altering column names
        return sm.OLS(y, X).fit()

    def __init__(self, X: ArrayLike, y: ArrayLike, model: Optional[RegressionResults]):
        self.X = X
        self.y = y
        if model:
            self.model = model
        else:
            self.fit_ols(X=X, y=y)

        self.residuals = self.model.fittedvalues

    def check_linearity(self) -> bool:
        return self.model.rsquared > 0

    def check_residuals_homoscedasticity(self) -> bool:
        _, p_value, _ = het_goldfeldquandt(self.residuals, self.X)
        return p_value > 0.05

    def check_residuals_normality(self) -> bool:
        if len(self.X) < 1000:
            # Use Shapiro-Wilk test
            _, p_value = shapiro(self.residuals)
        else:
            # Use Lilliefors test (Kolmogorov-Smirnov variant)
            _, p_value = lilliefors(self.residuals)
        return p_value > 0.5

    def check_residuals_autocorrelation(self) -> bool:
        dw_stat = durbin_watson(self.residuals)

        return 1.5 < dw_stat < 2.5

    def check_no_colinearity(self) -> bool:
        correlation_matrix = np.corrcoef(self.X, rowvar=False)
        if isinstance(correlation_matrix, float):  # single column
            return True
        else:
            coef_to_check = correlation_matrix[
                np.triu_indices_from(correlation_matrix, k=0)
            ]

            threshold = 0.8

            no_colinearity = all([corr < threshold for corr in coef_to_check.flat])
        return no_colinearity


@dataclass
class HypothesisCheckerResults:
    linearity: bool
    residuals_normality: bool
    residuals_homoscedasticity: bool
    residuals_no_autocorrelation: bool
    features_no_multicolinearity: bool

    @staticmethod
    def __from_given_input__(
        X: ArrayLike, y: ArrayLike, model: RegressionResults
    ) -> "HypothesisCheckerResults":

        hypothesis_checker = HypothesisChecker(X=X, y=y, model=model)

        return HypothesisCheckerResults(
            linearity=hypothesis_checker.check_linearity(),
            residuals_normality=hypothesis_checker.check_residuals_normality(),
            residuals_homoscedasticity=hypothesis_checker.check_residuals_normality(),
            residuals_no_autocorrelation=hypothesis_checker.check_residuals_autocorrelation(),
            features_no_multicolinearity=hypothesis_checker.check_no_colinearity(),
        )

    def __to_string__(self):
        return f"""--- Hypothesis check report ---
- Linearity: {self.linearity}
- Normality of the residuals: {self.residuals_normality}
- Homoscedasticity of the residuals: {self.residuals_homoscedasticity}
- No autocorrelation of the residuals: {self.residuals_no_autocorrelation}
- No multicolinearity in the features: {self.features_no_multicolinearity}
"""

    def get_check_report(self) -> str:
        return self.__to_string__()

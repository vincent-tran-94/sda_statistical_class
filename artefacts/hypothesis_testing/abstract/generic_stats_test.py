from abc import abstractmethod
from typing import Optional

from numpy._typing import ArrayLike
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.interface.interface_stats_test import (
    InterfaceStatsTest,
)


class GenericStatsTest(InterfaceStatsTest):
    """
    Base class for the statistical tests.
    """

    null_hypothesis: str

    def __init__(self):
        self._is_null_hypothesis_true: bool | None = None

    @abstractmethod
    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, threshold: float = 0.05
    ) -> "InterfaceStatsTest":
        """
        Used to compute the test statistic and p-value.
        Should update the "is_null_hypothesis_true" attribute.

        X (ArrayLike): The first sample.
        y (Optional[ArrayLike]): The second sample. If None, the test is a one-sample test.
        threshold (float): The threshold to reject the null hypothesis.
        """
        return self

    @property
    def is_null_hypothesis_true(self) -> bool:
        if self._is_null_hypothesis_true is not None:
            return self._is_null_hypothesis_true
        raise NotFittedError(
            "Test must be fitted before calling 'is_null_hypothesis_true'"
        )

import logging
from typing import Tuple

from pandas import DataFrame, Series

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


class TargetSplitterPreprocessor(PreprocessorInterface):
    def __init__(self):
        self.target_column: str = ""

    def fit(self, X: DataFrame, target_column: str) -> "TargetSplitterPreprocessor":
        self.target_column = target_column
        return self

    def transform(self, X: DataFrame) -> Tuple[DataFrame, Series]:
        logger = logging.getLogger("preprocessing_logger")
        logger.debug(f"Splitting target {self.target_column}")
        y = Series(data=X.loc[:, self.target_column], name=self.target_column)
        X = X.drop(columns=[self.target_column])
        return X, y

    def fit_transform(
        self, X: DataFrame, target_column: str
    ) -> Tuple[DataFrame, Series]:
        self.fit(X, target_column)
        return self.transform(X)

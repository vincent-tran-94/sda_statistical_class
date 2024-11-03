import logging

from pandas import DataFrame
from sklearn.exceptions import NotFittedError

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


class ColumnDropperPreprocessor(PreprocessorInterface):

    def __init__(self):
        self.column_name = None

    def fit(self, X: DataFrame, column_name: str) -> "ColumnDropperPreprocessor":
        self.column_name = column_name
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self.column_name:
            raise NotFittedError(
                "Column name not provided. Please fit the preprocessor first."
            )
        logger = logging.getLogger("preprocessing_logger")
        logger.debug(f"Dropping column {self.column_name}")
        return X.drop(labels=[self.column_name], axis=1)

    def fit_transform(self, X: DataFrame, column_name: str) -> DataFrame:
        return self.fit(X, column_name=column_name).transform(X)

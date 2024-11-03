import logging
from typing import Optional

from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


class QualitativeColumnsOneHotEncodingPreprocessor(PreprocessorInterface):
    def __init__(self):
        self.encoder = None
        self.columns = None

    def fit(
        self, X: DataFrame, columns: Optional[list[str]] = None
    ) -> "QualitativeColumnsOneHotEncodingPreprocessor":
        if not columns:
            self.columns = X.select_dtypes(include=["object"]).columns.tolist()
        else:
            self.columns = columns

        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X=X.loc[:, self.columns])

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        logger = logging.getLogger("preprocessing_logger")
        logger.debug(f"One hot encoding columns {self.columns} rows with NaN values")

        encoded_columns = self.encoder.transform(X.loc[:, self.columns])
        encoded_df = DataFrame(
            encoded_columns,
            columns=self.encoder.get_feature_names_out(self.columns),
            index=X.index,
        )
        X_transformed = X.drop(columns=self.columns).join(encoded_df)
        return X_transformed

    def fit_transform(self, X: DataFrame, columns: Optional[list[str]] = None):
        self.fit(X=X, columns=columns)
        return self.transform(X=X)

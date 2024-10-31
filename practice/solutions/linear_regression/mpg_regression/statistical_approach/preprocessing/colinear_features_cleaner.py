import logging

import numpy as np
from pandas import DataFrame


class ColinearFeatureCleanerPreprocessor:
    def __init__(self):
        self.to_drop: list[str] = None

    def fit(
        self, X: DataFrame, target_column: str, threshold: float = 0.8
    ) -> "ColinearFeatureCleanerPreprocessor":
        # Calculate the correlation matrix
        numeric_features = X.select_dtypes(include=np.number)

        corr_matrix = numeric_features.corr().abs()

        # Initialize an empty list to hold columns to drop
        to_drop = []

        for column in corr_matrix.columns:
            if column == target_column:
                continue
            else:
                for column_compared in corr_matrix.columns:
                    if column == target_column or column == column_compared:
                        continue
                    else:
                        if corr_matrix.loc[column_compared, column] > threshold:
                            to_drop.append(column)

        # Remove duplicates from the drop list
        self.to_drop = list(set(to_drop))
        logger = logging.getLogger("preprocessing_logger")
        logger.debug(msg=f"Collinear features to drop: {self.to_drop}")

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        # Drop the identified collinear features from X
        return X.drop(columns=self.to_drop, errors="ignore")

    def fit_transform(
        self, X: DataFrame, target_column: str, threshold: float = 0.9
    ) -> DataFrame:
        self.fit(X, threshold=threshold, target_column=target_column)
        return self.transform(X)

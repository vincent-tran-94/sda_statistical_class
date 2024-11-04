from typing import Tuple

from pandas import concat, DataFrame

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.colinear_features_cleaner import (
    ColinearFeatureCleanerPreprocessor,
)
from practice.solutions.logistic_regression.utils.preprocessing import (
    StatisticallySignificantFeatureSelector,
)
from practice.solutions.logistic_regression.utils.utils import load_breast_cancer_data


def preprocess_data(X: DataFrame, y: DataFrame) -> Tuple[DataFrame, DataFrame]:
    significant_features = StatisticallySignificantFeatureSelector().fit_transform(
        X=X, y=y
    )
    significant_features = (
        ColinearFeatureCleanerPreprocessor()
        .fit_transform(
            X=concat([significant_features, y], axis=1),
            target_column="has_cancer",
            threshold=0.9,
        )
        .drop(columns=["has_cancer"])
    )  # todo : refacto this
    return significant_features, y


if __name__ == "__main__":
    X, y = load_breast_cancer_data()
    significant_features, y = preprocess_data(X, y)
    print("Significant (non colinear) features:")
    print(
        f"Sur {X.shape[1]:.0f} variables, {significant_features.shape[1]:.0f} ont été identifiées comme significatives et non colinéaires."
    )
    print("- " + "\n- ".join(significant_features.columns.tolist()))

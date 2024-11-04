from typing import Tuple

from pandas import DataFrame, concat
from sklearn.datasets import load_breast_cancer

from practice.solutions.linear_regression.mpg_regression.analyse_exploratoire import (
    QuantitativeAnalysis,
)


def load_breast_cancer_data() -> Tuple[DataFrame, DataFrame]:
    data = load_breast_cancer()
    X = DataFrame(data.data, columns=data.feature_names)
    y = DataFrame(data.target, columns=["has_cancer"])
    return X, y


def pairplots_on_features(
    features: DataFrame, y: DataFrame, colors: list[str], features_per_plot: int = 9
):
    """
    Make arrays of 9 features and plot pairplots on them.
    """
    splits_with_target = []
    for i in range(0, features.shape[1], features_per_plot):
        split = features.iloc[:, i : min(i + features_per_plot, features.shape[1])]
        split_with_target = concat([split, y], axis=1)
        splits_with_target.append(split_with_target)

    for split in splits_with_target:
        QuantitativeAnalysis.plot_pairplot(data=split, colors=colors, hue="has_cancer")

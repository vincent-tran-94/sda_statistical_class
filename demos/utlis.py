from enum import Enum
from typing import Optional

from matplotlib import pyplot as plt
from numpy._typing import ArrayLike
from sklearn.datasets import make_regression, make_classification


class ProblemType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


def setup_plot():
    from aquarel import load_theme
    import seaborn as sns

    theme = load_theme("minimal_light")
    theme.apply()

    return ["#004aad", "#2bb4d4", "#2e2e2e", "#5ce1e6"]


def get_data(type: ProblemType):
    if type == ProblemType.REGRESSION:
        return make_regression(n_samples=200, n_features=1, bias=4, noise=20.0)
    else:
        return make_classification(
            n_samples=200,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            n_clusters_per_class=1,
        )


def plot_data(
    X: ArrayLike, y: ArrayLike, colors: list[str], preds: Optional[ArrayLike] = None
):
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, color=colors[0], alpha=0.6, label="Données")
    plt.xlabel("X")
    plt.ylabel("y")
    if preds is not None:
        plt.plot(X, preds, color=colors[1], label="Prediction")
        plt.title("Prédiction du modèle")
    else:
        plt.title("Target vs features")

    plt.legend()
    plt.grid()
    plt.show()

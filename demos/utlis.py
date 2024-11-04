from enum import Enum
from typing import Optional

import numpy as np
import statsmodels.api as sm

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy._typing import ArrayLike
from sklearn.datasets import make_regression, make_classification
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


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
            n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42
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


def plot_classification(
    X: ArrayLike,
    y: ArrayLike,
    colors: list[str],
    clf: Optional[BinaryResultsWrapper] = None,
):
    cmap_background = ListedColormap([colors[0], colors[1]])
    cmap_points = ListedColormap([colors[0], colors[1]])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    if clf:
        # Flatten the meshgrid arrays and add a column of ones for the intercept
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_with_intercept = sm.add_constant(grid)

        # Predict and reshape to match the shape of the meshgrid
        Z = clf.predict(grid_with_intercept)
        Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    if clf:
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap_background)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor=colors[2], s=50)
    if clf:
        plt.title(
            "Frontière de décision de la régression logistique", fontweight="bold"
        )
    else:
        plt.title("Visualisation du dataset", fontweight="bold")
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.grid(True)
    plt.show()

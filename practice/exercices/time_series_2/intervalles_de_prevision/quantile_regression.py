import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from demos.utlis import setup_plot

if __name__ == "__main__":
    colors = setup_plot()

    np.random.seed(42)
    n_samples = 500
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.flatten() + np.sin(X.flatten()) * 10 + np.random.normal(0, 2, n_samples)
    # todo : BONUS : Faire une régression quantile en utilisant X commes variables d'entrée et
    # todo : y comme vecteur de résultat

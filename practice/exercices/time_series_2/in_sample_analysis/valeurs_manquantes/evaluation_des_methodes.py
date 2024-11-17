import numpy as np
from pandas import Series, DataFrame
from sklearn.metrics import mean_squared_error

from demos.utlis import get_time_series_with_missing_values

if __name__ == "__main__":
    data, _ = get_time_series_with_missing_values()
    # todo : implémenter une autre méthode d'imputation des valeurs manquantes
    # todo : déterminer quelle est la meilleure méthode d'imputation sur ce jeu de données

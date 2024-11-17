import math

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

from demos.utlis import setup_plot, get_time_series_with_missing_values


if __name__ == "__main__":
    data, null_indexes = get_time_series_with_missing_values()
    colors = setup_plot()
    # todo: proposer un algorithme d'imputation des valeurs manquantes (polynome d'interpolation, moyenne mobile robuste, algorithme de ML, Filtre de Kalman par SARIMAX, etc.)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX

from demos.utlis import setup_plot, get_time_series_with_outliers


if __name__ == "__main__":
    data = get_time_series_with_outliers()
    colors = setup_plot()
    # todo: proposer un algorithme de détection des valeurs aberrantes (vu en cours, Isolation Forest, etc)

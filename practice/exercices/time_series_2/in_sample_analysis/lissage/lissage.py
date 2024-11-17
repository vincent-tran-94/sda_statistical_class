from matplotlib import pyplot as plt
from pandas import Series
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from demos.utlis import setup_plot, get_noisy_time_series

if __name__ == "__main__":
    data = get_noisy_time_series()
    colors = setup_plot()
    # todo: utiliser et afficher les méthodes de lissage suivantes: moyenne mobile, médiane mobile, lissage exponentiel

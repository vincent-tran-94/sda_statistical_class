import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from pandas import Series
from demos.utlis import setup_plot, get_noisy_time_series

if __name__ == "__main__":
    data = get_noisy_time_series()
    colors = setup_plot()
    # todo: utiliser et afficher les filtres suivants: moyenne mobile (par produit de convolution), filtre gaussien, filtre de Savitzky-Golay

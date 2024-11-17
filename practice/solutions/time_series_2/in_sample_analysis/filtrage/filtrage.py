import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from pandas import Series
from demos.utlis import setup_plot, get_noisy_time_series


def apply_moving_average(data: Series, window: int) -> Series:
    """
    Applies a moving average filter to a time series.

    Parameters:
    data (Series): The time series data.
    window (int): The rolling window size.

    Returns:
    Series: The moving average of the series.
    """
    return Series(
        np.convolve(data, np.ones(window) / window, mode="same"),
        index=data.index,
        name=f"moving_average_window_{window}",
    )


def apply_gaussian_filter(data: Series, sigma: float) -> Series:
    """
    Applies a Gaussian filter to a time series.

    Parameters:
    data (Series): The time series data.
    sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
    Series: The Gaussian-smoothed series.
    """
    return Series(
        gaussian_filter1d(data, sigma=sigma),
        index=data.index,
        name=f"gaussian_filter_sigma_{sigma}",
    )


def apply_savgol_filter(data: Series, window: int, poly_order: int) -> Series:
    """
    Applies a Savitzky-Golay filter to a time series.

    Parameters:
    data (Series): The time series data.
    window (int): The window length for the filter.
    poly_order (int): The order of the polynomial to fit.

    Returns:
    Series: The Savitzky-Golay smoothed series.
    """
    return Series(
        savgol_filter(data, window_length=window, polyorder=poly_order),
        index=data.index,
        name=f"savgol_window_{window}_poly_{poly_order}",
    )


def plot_moving_average(
    ax, data: Series, windows: list[int], colors: list[str]
) -> None:
    """
    Plots moving average filtered series for various window sizes.

    Parameters:
    ax (list[plt.Axes]): List of axes for plotting.
    data (Series): The original time series data.
    windows (list[int]): List of window sizes for the moving average.
    colors (list[str]): List of colors for plot lines.
    """
    for i, window in enumerate(windows):
        moving_avg = apply_moving_average(data=data, window=window)
        ax[i].plot(data, label="Raw Data", color=colors[0], alpha=0.7)
        ax[i].plot(moving_avg, label=f"Window: {window}", color=colors[2])
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Moving Average")


def plot_gaussian_filter(
    ax, data: Series, sigmas: list[float], colors: list[str]
) -> None:
    """
    Plots Gaussian-filtered series for various sigma values.

    Parameters:
    ax (list[plt.Axes]): List of axes for plotting.
    data (Series): The original time series data.
    sigmas (list[float]): List of standard deviations (sigma) for Gaussian smoothing.
    colors (list[str]): List of colors for plot lines.
    """
    for i, sigma in enumerate(sigmas):
        gaussian_smooth = apply_gaussian_filter(data=data, sigma=sigma)
        ax[i].plot(data, label="Raw Data", color=colors[0], alpha=0.7)
        ax[i].plot(gaussian_smooth, label=f"Sigma: {sigma}", color=colors[2])
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Gaussian Filter")


def plot_savgol_filter(
    ax, data: Series, windows: list[int], poly_orders: list[int], colors: list[str]
) -> None:
    """
    Plots Savitzky-Golay filtered series for various window and polynomial order combinations.

    Parameters:
    ax (list[plt.Axes]): List of axes for plotting.
    data (Series): The original time series data.
    windows (list[int]): List of window lengths for the Savitzky-Golay filter.
    poly_orders (list[int]): List of polynomial orders for the Savitzky-Golay filter.
    colors (list[str]): List of colors for plot lines.
    """
    for i, (window, poly_order) in enumerate(zip(windows, poly_orders)):
        savgol_smooth = apply_savgol_filter(
            data=data, window=window, poly_order=poly_order
        )
        ax[i].plot(data, label="Raw Data", color=colors[0], alpha=0.7)
        ax[i].plot(
            savgol_smooth,
            label=f"Window: {window}, Poly: {poly_order}",
            color=colors[2],
        )
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Savitzky-Golay Filter")


if __name__ == "__main__":
    data = get_noisy_time_series()
    colors = setup_plot()
    # todo: utiliser et afficher les filtres suivants: moyenne mobile (par produit de convolution), filtre gaussien, filtre de Savitzky-Golay

    # Parameters for the filters
    moving_avg_windows = [12, 24, 36]
    gaussian_sigmas = [2, 5, 10]
    savgol_windows = [11, 21, 31]  # Must be odd
    savgol_poly_orders = [2, 3, 4]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharey=True, sharex=True)

    # Apply and plot each filter with various parameters
    plot_moving_average(ax[0], data=data, windows=moving_avg_windows, colors=colors)
    plot_gaussian_filter(ax[1], data=data, sigmas=gaussian_sigmas, colors=colors)
    plot_savgol_filter(
        ax[2],
        data=data,
        windows=savgol_windows,
        poly_orders=savgol_poly_orders,
        colors=colors,
    )

    # Set x-tick frequency
    for row in ax:
        for col in row:
            col.xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    plt.show()

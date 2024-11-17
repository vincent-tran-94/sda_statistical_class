from matplotlib import pyplot as plt
from pandas import Series
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from demos.utlis import setup_plot, get_noisy_time_series


def compute_moving_average(data: Series, window: int) -> Series:
    """
    Computes the moving average of a time series.

    Parameters:
    data (DataFrame): The time series data.
    window (int): The rolling window size.

    Returns:
    DataFrame: The moving average of the series.
    """
    return data.rolling(window=window).mean()


def compute_moving_median(data: Series, window: int) -> Series:
    """
    Computes the moving median of a time series.

    Parameters:
    data (DataFrame): The time series data.
    window (int): The rolling window size.

    Returns:
    DataFrame: The moving median of the series.
    """
    return data.rolling(window=window).median()


def compute_exponential_smoothing(data: Series, alpha: float) -> Series:
    """
    Applies exponential smoothing to a time series.

    Parameters:
    data (DataFrame): The time series data.
    alpha (float): The smoothing factor.

    Returns:
    DataFrame: The exponentially smoothed series.
    """
    smoother = ExponentialSmoothing(
        data, trend="add", seasonal="add", seasonal_periods=12
    ).fit(smoothing_level=alpha)
    return smoother.predict(start=1, end=len(data))


def plot_moving_average(
    ax, data: Series, windows: list[int], colors: list[str]
) -> None:
    """
    Plots moving averages for various window sizes.

    Parameters:
    ax (list[plt.Axes]): Axes for plotting.
    data (DataFrame): The time series data.
    windows (list[int]): List of rolling window sizes.
    colors (list[str]): Colors for plotting.
    """
    for i, window in enumerate(windows):
        moving_avg = compute_moving_average(data=data, window=window)
        ax[i].plot(data, label="Raw Data", color=colors[0], alpha=0.7)
        ax[i].plot(moving_avg, label=f"Window: {window}", color=colors[2])
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Moving Average")


def plot_moving_median(ax, data: Series, windows: list[int], colors: list[str]) -> None:
    """
    Plots moving medians for various window sizes.

    Parameters:
    ax (list[plt.Axes]): Axes for plotting.
    data (DataFrame): The time series data.
    windows (list[int]): List of rolling window sizes.
    colors (list[str]): Colors for plotting.
    """
    for i, window in enumerate(windows):
        moving_median = compute_moving_median(data=data, window=window)
        ax[i].plot(data, label="Raw Data", color=colors[0], alpha=0.7)
        ax[i].plot(moving_median, label=f"Window: {window}", color=colors[2])
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Moving Median")


def plot_exponential_smoothing(
    ax, data: Series, alphas: list[float], colors: list[str]
) -> None:
    """
    Plots exponential smoothing for various alpha values.

    Parameters:
    ax (list[plt.Axes]): Axes for plotting.
    data (DataFrame): The time series data.
    alphas (list[float]): List of smoothing factors (alpha values).
    colors (list[str]): Colors for plotting.
    """
    for i, alpha in enumerate(alphas):
        exp_smoothing = compute_exponential_smoothing(data=data, alpha=alpha)
        ax[i].plot(data, label="Raw Data", color=colors[0])
        ax[i].plot(exp_smoothing, label=f"Alpha: {alpha}", color=colors[2])
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_ylabel("Exponential Smoothing")


if __name__ == "__main__":
    # todo: utiliser et afficher les méthodes de lissage suivantes: moyenne mobile, médiane mobile, lissage exponentiel
    data = get_noisy_time_series()
    colors = setup_plot()
    rolling_windows = [12 * i for i in range(1, 4)]
    alphas = [0.1, 0.5, 0.9]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharey=True, sharex=True)
    plot_moving_average(ax[0], data=data, windows=rolling_windows, colors=colors)
    plot_moving_median(ax[1], data=data, windows=rolling_windows, colors=colors)
    plot_exponential_smoothing(ax[2], data=data, alphas=alphas, colors=colors)

    for row in ax:
        for col in row:
            col.xaxis.set_major_locator(
                plt.MaxNLocator(5)
            )  # Limits number of x-ticks to 5

    plt.tight_layout()
    plt.show()

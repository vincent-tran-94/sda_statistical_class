import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX

from demos.utlis import setup_plot, get_time_series_with_outliers


def get_residuals(data: Series, prediction: Series) -> Series:
    """
    Get the residuals of the method.
    Args:
        data (Series): The time series data.
        prediction (Series): The prediction of the model.

    Returns:
        Series: The residuals of the model.
    """
    return data - prediction


def detect_outliers(residuals: Series) -> Series:
    """
    Detects outliers in a time series.

    Parameters:
    data (Series): The time series data.
    residuals (Series): The residuals of the model.

    Returns:
    Series: A boolean series indicating the outliers.
    """
    # Assuming residuals are gaussian, we can use the 3-sigma rule
    return residuals.abs() > 3 * residuals.std()


def get_insample_prediction(data: Series, model: SARIMAXResults) -> Series:
    """
    Get the in-sample prediction of the model.
    Args:
        data (Series): The time series data.

    Returns:
        Series: The in-sample prediction of the model.
    """
    return model.predict(start=1, end=len(data))


def fit_model(
    data: Series,
    p: int = 2,
    d: int = 1,
    q: int = 1,
    P: int = 1,
    D: int = 1,
    Q: int = 1,
    seasonal_frequency: int = 12,
) -> SARIMAXResults:
    """
    Fits a model to a time series.

    Parameters:
    data (Series): The time series data.
    p (int): The AR order.
    d (int): The differencing order.
    q (int): The MA order.
    P (int): The seasonal AR order.
    D (int): The seasonal differencing order.
    Q (int): The seasonal MA order.
    seasonal_frequency (int): The frequency of the seasonality.

    Returns:
    Series: The model fit to the data.
    """
    return SARIMAX(
        data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, seasonal_frequency),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()


def plot_outliers(
    data: Series,
    in_sample_prediction: Series,
    residuals: Series,
    outliers: Series,
    colors: list[str],
) -> None:
    """
    Plots the time series data with outliers highlighted.

    Parameters:
    data (Series): The time series data.
    in_sample_prediction (Series): The in-sample prediction of the model.
    residuals (Series): The residuals of the model.
    outliers (Series): A boolean series indicating the outliers.
    colors (list[str]): The colors to use for plotting.
    """

    fig, axes = plt.subplots(figsize=(15, 8), nrows=2, ncols=1, sharex=False)

    # Plot time series, moving average, and error points on primary y-axis
    axes[0].plot(data.index, data, label="Série Temporelle", color=colors[0])
    axes[0].plot(
        data.index,
        in_sample_prediction,
        label="Filtre de Kalman",
        color=colors[1],
    )

    errors_idx = data[outliers].index

    axes[0].scatter(errors_idx, data[errors_idx], color=colors[2], label="Erreurs")

    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    sns.kdeplot(
        residuals,
        color=colors[3],
        label="Distribution des erreurs",
        fill=True,
        ax=axes[1],
    )
    axes[1].axvline(x=-3 * np.std(residuals), color=colors[2], linestyle="--")
    axes[1].axvline(x=3 * np.std(residuals), color=colors[2], linestyle="--")
    axes[1].set_ylabel("Densité des erreurs")

    # Adjust legend for both plots
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.suptitle(
        "Détection des anomalies par le filtre de Kalman (SARIMA)", fontweight="bold"
    )
    plt.show()


if __name__ == "__main__":
    data = get_time_series_with_outliers()
    colors = setup_plot()
    # todo: proposer un algorithme de détection des valeurs aberrantes (vu en cours, Isolation Forest, etc)

    fitted_model = fit_model(data=data)
    insample_prediction = get_insample_prediction(data=data, model=fitted_model)
    residuals = get_residuals(data=data, prediction=insample_prediction)
    outliers = detect_outliers(residuals=residuals)

    plot_outliers(
        data=data,
        in_sample_prediction=insample_prediction,
        residuals=residuals,
        outliers=outliers,
        colors=colors,
    )

from typing import Tuple

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.collections import PolyCollection, LineCollection


def custom_acf(ax, serie, lags: int, color: str):
    plot_acf(serie, lags=lags, color=color, ax=ax)

    # Customize the ACF plot colors
    for item in ax.collections:
        # Change the color of the confidence intervals
        if isinstance(item, PolyCollection):
            item.set_facecolor(color)
        # Change the color of the vertical lines
        if isinstance(item, LineCollection):
            item.set_color(color)

            # Change the color of the markers/horizontal line
    for item in ax.lines:
        item.set_color(color)

    ax.grid(True)


def custom_pacf(ax, serie, lags: int, color: str):
    plot_pacf(serie, lags=lags, color=color, ax=ax)

    # Customize the PACF plot colors
    for item in ax.collections:
        # Change the color of the confidence intervals
        if isinstance(item, PolyCollection):
            item.set_facecolor(color)
        # Change the color of the vertical lines
        if isinstance(item, LineCollection):
            item.set_color(color)

            # Change the color of the markers/horizontal line
    for item in ax.lines:
        item.set_color(color)

    ax.grid(True)


def plot_acf_pacf(
    serie,
    color: str,
    lags: int = 20,
    save_path: str | None = None,
    figsize: Tuple[int, int] = (10, 12),
):
    fig, axs = plt.subplots(2, 1, figsize=figsize)  # Create a 2x1 grid of subplots

    # Plot ACF in the first subplot
    custom_acf(axs[0], serie=serie, lags=lags, color=color)
    axs[0].set_title("Autocorrelation Function (ACF)")

    # Plot PACF in the second subplot
    custom_pacf(axs[1], serie=serie, lags=lags, color=color)
    axs[1].set_title("Partial Autocorrelation Function (PACF)")

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

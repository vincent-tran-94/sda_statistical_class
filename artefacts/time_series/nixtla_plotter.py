import pandas as pd
from matplotlib import pyplot as plt

from demos.utlis import setup_plot


def plot_cross_validation_results(crossvalidation_df: pd.DataFrame, cutoff: list):
    num_cutoffs = len(cutoff)
    colors = setup_plot()
    ncols = 2  # Set to 2 or any suitable number of columns
    nrows = (num_cutoffs + ncols - 1) // ncols  # Calculate the required number of rows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 6 * nrows))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Loop through each cutoff and plot on each subplot
    for k in range(num_cutoffs):
        cv = crossvalidation_df[crossvalidation_df.loc[:, "cutoff"] == cutoff[k]]
        ax = axes[k]  # Select the subplot based on the current loop index

        # Plot the actual values and model predictions
        ax.plot(cv["ds"], cv["actual"], label="actual", color=colors[0])
        for model in ["AutoETS", "AutoARIMA", "Holt", "HoltWinters", "AutoTBATS"]:
            ax.plot(cv["ds"], cv[model], label=model)

        ax.legend()
        ax.grid()

    # Hide any unused subplots (in case nrows*ncols > num_cutoffs)
    for k in range(num_cutoffs, len(axes)):
        axes[k].axis("off")

    # Show the combined figure
    plt.tight_layout()
    plt.show()

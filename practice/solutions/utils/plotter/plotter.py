# imports
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
from aquarel import load_theme
from numpy._typing import ArrayLike

# setup
theme = load_theme("minimal_light")
theme.apply()
COLORS = ["#004aad", "#2bb4d4", "#2e2e2e", "#5ce1e6"]


def plot_two_samples(
    sample_1: ArrayLike,
    sample_2: ArrayLike,
    sample_1_label: str = "Sample 1",
    sample_2_label: str = "Sample 2",
    figure_title: str = "Two samples comparison",
    x_label: str = "X",
    alternative_plot: Literal["violin", "ecdf", "scatter"] = "violin",
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # KDE plot for distribution comparison
    sns.kdeplot(
        sample_1,
        fill=True,
        color=COLORS[0],
        label=sample_1_label,
        ax=axes[0],
        alpha=0.5,
    )
    sns.kdeplot(
        sample_2,
        fill=True,
        color=COLORS[1],
        label=sample_2_label,
        ax=axes[0],
        alpha=0.5,
    )
    axes[0].set_xlabel(x_label)
    axes[0].legend()
    axes[0].grid(True)

    # Alternative plot selection
    if alternative_plot == "violin":
        violins = axes[1].violinplot(
            [sample_1, sample_2], vert=True, showmeans=False, showmedians=True
        )
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(
            [sample_1_label, sample_2_label], color="black", fontweight="bold"
        )
        for i, pc in enumerate(violins["bodies"]):
            pc.set_facecolor(COLORS[i])
            pc.set_edgecolor("white")
        violins["cmedians"].set_color("white")  # Set median color
        violins["cmins"].set_color("white")  # Set min whisker color
        violins["cmaxes"].set_color("white")  # Set max whisker color

    elif alternative_plot == "ecdf":
        sns.ecdfplot(
            sample_1, color=COLORS[0], alpha=0.5, label=sample_1_label, ax=axes[1]
        )
        sns.ecdfplot(
            sample_2, color=COLORS[1], alpha=0.5, label=sample_2_label, ax=axes[1]
        )
        axes[1].set_xlabel(x_label)

    elif alternative_plot == "scatter":
        sns.scatterplot(x=sample_1, y=sample_2, ax=axes[1], color=COLORS[2])
        axes[1].set_xlabel(sample_1_label)
        axes[1].set_ylabel(sample_2_label)

    else:
        raise ValueError("Invalid alternative plot type: choose 'violin' or 'ecdf'")

    axes[1].legend()
    axes[1].grid(True)
    plt.suptitle(figure_title, fontweight="bold")

    plt.show()

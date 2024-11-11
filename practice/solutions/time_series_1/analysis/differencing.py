from demos.utlis import get_time_series, setup_plot
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    # Create a 2x1 grid of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the CO₂ time series on the first subplot
    ax1.plot(data.index, data.co2, color=colors[0])
    ax1.set_title("CO₂ Levels Over Time (Monthly)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CO₂ Levels")
    ax1.grid(True)

    differentiated_serie = data.co2.diff().dropna()
    # Plot the KDE of CO₂ levels on the second subplot
    ax2.plot(differentiated_serie.index, differentiated_serie, color=colors[0])
    ax2.set_title("CO₂ Differentiated - (Stationary)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CO₂ Levels")
    ax2.grid(True)

    plt.tight_layout()  # Adjust spacing between plots

    plt.show()

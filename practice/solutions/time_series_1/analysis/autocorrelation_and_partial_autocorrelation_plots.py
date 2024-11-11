from artefacts.hypothesis_testing.time_series.plots import plot_acf_pacf
from demos.utlis import get_time_series, setup_plot

if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    plot_acf_pacf(data.co2[-12 * 10 :], lags=31, color=colors[0], figsize=(8, 10))

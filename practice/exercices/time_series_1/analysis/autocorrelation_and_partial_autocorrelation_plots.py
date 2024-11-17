from artefacts.time_series.plots import plot_acf_pacf
from demos.utlis import get_time_series, setup_plot

if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()
    # todo : plot Acf and Pacf

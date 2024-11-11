import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from artefacts.hypothesis_testing.time_series.plots import plot_acf_pacf
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast.models import ARIMA as sf_arima
from statsforecast import StatsForecast


if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = (
        data.iloc[-500:, :].diff().dropna(axis=0)
    )  # 1st order diff to make it stationnary
    # fit an MA model and predict

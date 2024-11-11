import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from artefacts.hypothesis_testing.time_series.plots import plot_acf_pacf
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast.models import ARIMA
from statsforecast import StatsForecast


if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = data.iloc[-500:, :]  # We do not need to difference the data anymore
    # fit an SARIMA(X) model and predict

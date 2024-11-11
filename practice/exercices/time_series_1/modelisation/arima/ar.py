import pandas as pd
from matplotlib import pyplot as plt
from statsforecast.models import AutoRegressive
from statsmodels.tsa.ar_model import AutoReg

from artefacts.hypothesis_testing.time_series.plots import plot_acf_pacf
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast import StatsForecast


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    modelled_data = (
        data.iloc[-500:, :].diff().dropna(axis=0)
    )  # 1st order diff to make it stationnary

    # fit an AR model and predict

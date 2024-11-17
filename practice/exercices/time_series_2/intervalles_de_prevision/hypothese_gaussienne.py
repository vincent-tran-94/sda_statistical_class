import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
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

    # todo : établir des intervalles de prévisions en supposant la normalité des résidus
    # todo : Pour ce faire, il faut utiliser la méthode de prévision des intervalles de confiance par défaut de statsmodels & statsforecast

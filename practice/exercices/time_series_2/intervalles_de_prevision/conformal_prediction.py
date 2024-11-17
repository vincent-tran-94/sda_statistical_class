import matplotlib.pyplot as plt
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from statsforecast.utils import ConformalIntervals

from demos.utlis import get_time_series, setup_plot, to_nixtla_format

from mlforecast import MLForecast

from window_ops.expanding import expanding_mean


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    data = to_nixtla_format(data=data, target_name="co2")
    # todo : Etablir des intervalles de prévision avec les conformal prediction
    # todo : https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html

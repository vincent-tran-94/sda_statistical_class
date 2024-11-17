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


def plot_model_prediction(
    data: DataFrame,
    preds: DataFrame,
    model_name: str,
    levels: list[int],
    colors: list[int],
) -> None:

    plt.figure(figsize=(12, 8))
    plt.plot(data.ds, data.y, label="Actual", color=colors[0])
    plt.plot(
        preds.ds,
        preds.loc[:, model_name],
        label="Point forecasting",
        color=colors[2],
    )
    for level in levels:
        plt.fill_between(
            preds.ds,
            preds.loc[:, f"{model_name}-lo-{level}"],
            preds.loc[:, f"{model_name}-hi-{level}"],
            alpha=0.2 if level > 90 else 0.8,
            label=f"{level}% PI",
            color=colors[3],
        )
    plt.grid(True)
    plt.legend()
    plt.title(f"Prévision ponctuelle et intervalles de confiance - {model_name}")
    plt.show()


def get_machine_learning_forecast():
    horizon = 36
    seasonal_period = 12
    prediction_intervals = PredictionIntervals(n_windows=3, h=horizon)
    forecast_model = MLForecast(
        models=[GradientBoostingRegressor()],
        lags=[lag for lag in range(1, 4)],
        target_transforms=[Differences([1])],  # differentiation
        lag_transforms={
            1: [
                expanding_mean,
            ]
        },
        freq="W-SUN",
    ).fit(data, prediction_intervals=prediction_intervals)

    preds = forecast_model.predict(h=horizon, level=[80, 95])
    return preds


def get_statistical_forecast():
    horizon = 36
    seasonal_period = 12
    p = 4
    d = 1
    q = 1
    P = 1
    D = 1
    Q = 1
    prediction_intervals = ConformalIntervals(n_windows=3, h=horizon)
    stat_models = StatsForecast(
        models=[
            ARIMA(
                order=(p, d, q),
                seasonal_order=(P, D, Q),
                season_length=seasonal_period,
            )
        ],
        freq="W-SUN",
    ).fit(data, prediction_intervals=prediction_intervals)

    preds = stat_models.predict(h=horizon, level=[80, 95])
    return preds


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    data = to_nixtla_format(data=data, target_name="co2")
    # todo : mlforecast & statsforecast conformal pred

    preds = get_machine_learning_forecast()
    plot_model_prediction(
        data=data,
        colors=colors,
        preds=preds,
        model_name="GradientBoostingRegressor",
        levels=[80, 95],
    )

    preds = get_statistical_forecast()
    plot_model_prediction(
        data=data,
        colors=colors,
        preds=preds,
        model_name="ARIMA",
        levels=[80, 95],
    )

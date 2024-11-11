import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters, SeasonalExponentialSmoothingOptimized

from practice.solutions.time_series_1.modelisation.plots_ts_results import (
    plot_preds_results,
)


def statsmodels_modelisation(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    ses_model = ExponentialSmoothing(
        data.loc[:, target_name],
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    ).fit(smoothing_trend=0.8, smoothing_seasonal=0.8)
    ses_preds = ses_model.forecast(steps=12)
    return ses_preds


def statsforecast_modelisation(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    models = [
        HoltWinters(season_length=12),
        SeasonalExponentialSmoothingOptimized(
            season_length=12, alias="Seasonal Exponential Smoothing Optimized"
        ),
    ]

    df = to_nixtla_format(data=data, target_name=target_name)

    sf = StatsForecast(models=models, n_jobs=-1, freq="W-SUN")  # hourly frequency

    sf.fit(df=df)

    preds = sf.predict(h=12)

    return from_nixtla_format(preds)


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    statsmodels_preds = statsmodels_modelisation(data=data, target_name="co2")
    statsforecast_preds = statsforecast_modelisation(data=data, target_name="co2")

    plotted_data = data[-120:]

    plot_preds_results(
        data=data[-120:],
        statsmodels_preds=statsmodels_preds,
        statsforecast_preds=statsforecast_preds,
    )

import matplotlib.pyplot as plt
import numpy as np
from coreforecast.scalers import inv_boxcox
from statsforecast.tbats import tbats_selection, tbats_forecast
from statsforecast.utils import AirPassengers as ap

from demos.utlis import setup_plot

y = ap
seasonal_periods = np.array([12])

# Default parameters
use_boxcox = None
bc_lower_bound = 0
bc_upper_bound = 1
use_trend = None
use_damped_trend = None
use_arma_errors = True

model = tbats_selection(
    y,
    seasonal_periods,
    use_boxcox,
    bc_lower_bound,
    bc_upper_bound,
    use_trend,
    use_damped_trend,
    use_arma_errors,
)


if __name__ == "__main__":

    # https://nixtlaverse.nixtla.io/statsforecast/src/tbats.html#tbats-model

    horizon = 24
    fcst = tbats_forecast(mod=model, h=horizon)
    forecast = fcst["mean"]
    if model["BoxCox_lambda"] is not None:
        forecast = inv_boxcox(forecast, model["BoxCox_lambda"])

    colors = setup_plot()
    plt.figure(figsize=(12, 6))
    y_index = np.arange(len(y))
    plt.plot(y_index, y, color=colors[0], label="Original")
    plt.fill_between(x=y_index, y1=[0] * len(y), y2=y, color=colors[0], alpha=0.1)

    forecast_index = np.arange(len(y), len(y) + horizon)
    plt.plot(forecast_index, forecast, color=colors[1], label="Forecast")
    plt.fill_between(
        x=forecast_index,
        y1=[0] * len(forecast),
        y2=forecast,
        color=colors[1],
        alpha=0.1,
    )
    plt.legend()
    plt.grid(True)
    plt.title("Prévision du TBATS", fontweight="bold")
    plt.show()

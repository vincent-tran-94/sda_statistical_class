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
    # todo : fit a tbats and predict

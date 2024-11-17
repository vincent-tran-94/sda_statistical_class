from demos.utlis import (
    get_time_series,
    setup_plot,
)

if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = data.iloc[-500:, :]  # We do not need to difference the data anymore
    # todo : fit an ARIMA model and predict

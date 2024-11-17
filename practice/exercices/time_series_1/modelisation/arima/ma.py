from demos.utlis import (
    get_time_series,
    setup_plot,
)

if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = (
        data.iloc[-500:, :].diff().dropna(axis=0)
    )  # 1st order diff to make it stationnary
    # todo : fit an MA model and predict

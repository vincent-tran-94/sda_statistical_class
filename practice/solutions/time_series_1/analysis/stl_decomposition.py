from demos.utlis import get_time_series, setup_plot
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    # fitting stl
    stl = STL(data, seasonal=13)
    stl = stl.fit()

    # getting results
    observed = stl.observed
    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid

    # plotting results
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 8), sharex=True)
    axes[0].plot(data, color=colors[0], label="Observed")
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    axes[1].plot(trend, color=colors[1], label="Trend")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    axes[2].plot(seasonal, color=colors[2], label="Seasonal")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)

    axes[3].scatter(resid.index, resid, color=colors[3], label="Residual")
    axes[3].legend(loc="upper left")
    axes[3].grid(True)

    plt.legend(loc="upper left")
    plt.show()

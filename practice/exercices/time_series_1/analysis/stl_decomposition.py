from demos.utlis import get_time_series, setup_plot
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    # perform and plot stl decomposition

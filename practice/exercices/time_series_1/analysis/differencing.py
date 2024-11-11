from demos.utlis import get_time_series, setup_plot
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    # Plot the data in the temporal domain and the differentiated data

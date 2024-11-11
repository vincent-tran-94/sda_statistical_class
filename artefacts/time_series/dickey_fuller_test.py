import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


class AdvancedADFTest:
    def __init__(self, series, alpha=0.05):
        """
        Initializes the AdvancedADFTest class.

        Parameters:
            series (pd.Series): Time series data to be tested.
            alpha (float): Significance level for stationarity check. Default is 0.05.
        """
        self.series = series
        self.alpha = alpha
        self.result = None

    def run_test(self):
        """
        Performs the Augmented Dickey-Fuller test on the time series data.

        Returns:
            dict: A dictionary with test statistics and p-value.
        """
        self.result = adfuller(self.series, autolag="AIC")
        output = {
            "ADF Statistic": self.result[0],
            "p-value": self.result[1],
            "Critical Values": self.result[4],
        }
        return output

    def is_stationary(self):
        """
        Checks if the time series is stationary based on the ADF test p-value.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        if self.result is None:
            self.run_test()
        return self.result[1] < self.alpha

    def summary(self):
        """
        Provides a detailed summary of the ADF test results and stationarity check.

        Returns:
            str: A summary of the ADF test results.
        """
        if self.result is None:
            self.run_test()
        adf_stat = self.result[0]
        p_value = self.result[1]
        critical_values = self.result[4]

        summary_text = f"ADF Statistic: {adf_stat:.4f}\n"
        summary_text += f"p-value: {p_value:.4f}\n"
        summary_text += "Critical Values:\n"
        for key, value in critical_values.items():
            summary_text += f"   {key}: {value:.4f}\n"

        summary_text += "\n"
        if self.is_stationary():
            summary_text += f"The series is stationary (p < {self.alpha}).\n"
        else:
            summary_text += f"The series is not stationary (p >= {self.alpha}).\n"

        return summary_text

    def plot_series(self, color: str = "C0"):
        """
        Plots the time series data for visual inspection.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.series, label="Time Series", color=color)
        plt.title("Time Series Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

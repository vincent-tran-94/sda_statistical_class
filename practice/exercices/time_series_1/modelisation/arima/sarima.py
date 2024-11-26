from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

from demos.utlis import (
    get_time_series,
    setup_plot,
)

if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()


    modelled_data = data.iloc[-500:, :]  # We do not need to difference the data anymore
    # todo : fit an SARIMA(X) model and predict
    #AIC minimiser la valeur pour avoir une bonne performance du modèle 
    model = SARIMAX(
        endog=modelled_data, order=(3, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False,enforce_stationarity=False
    )

    fitted = model.fit()
    print("Stastitiques de coefficient",fitted.summary())
    preds = fitted.forecast(steps=12)
    print(preds.head())

    #print(modelled_data)
    
    plot_data = data.iloc[-50:, :]
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data.index, plot_data.co2, label="Actual", color=colors[0])
    plt.plot(preds.index, preds, label="Predicted", color=colors[1])
    plt.legend()
    plt.grid()
    plt.title("C02 ARIMA forecast")
    plt.show()
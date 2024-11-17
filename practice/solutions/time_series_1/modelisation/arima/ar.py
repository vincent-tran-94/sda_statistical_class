import pandas as pd
from statsforecast.models import AutoRegressive
from statsmodels.tsa.ar_model import AutoReg

from artefacts.time_series.plots import plot_acf_pacf
from artefacts.time_series.dickey_fuller_test import AdvancedADFTest
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast import StatsForecast

from practice.solutions.time_series_1.modelisation.plots_ts_results import (
    plot_preds_results,
)


def statsmodels_modelisation(
    data: pd.DataFrame, target_name: str, p: int = 1
) -> pd.DataFrame:
    ses_model = AutoReg(data.loc[:, target_name], lags=p).fit()
    ses_preds = ses_model.forecast(steps=12)
    return ses_preds


def statsforecast_modelisation(
    data: pd.DataFrame, target_name: str, p: int = 1
) -> pd.DataFrame:
    models = [
        AutoRegressive(lags=p),
    ]

    df = to_nixtla_format(data=data, target_name=target_name)

    sf = StatsForecast(models=models, n_jobs=-1, freq="W-SUN")  # hourly frequency

    sf.fit(df=df)

    preds = sf.predict(h=12)

    return from_nixtla_format(preds)


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    modelled_data = (
        data.iloc[-500:, :].diff().dropna(axis=0)
    )  # 1st order diff to make it stationnary

    # run dickey fuller test
    df_test = AdvancedADFTest(series=modelled_data.co2)
    df_test.run_test()
    df_test.plot_series(color=colors[0])
    print(df_test.summary())

    statsmodels_preds = statsmodels_modelisation(
        data=modelled_data, target_name="co2", p=5
    )
    statsforecast_preds = statsforecast_modelisation(
        data=modelled_data, target_name="co2", p=5
    )

    plotted_data = modelled_data[-120:]

    # acf/pacf
    plot_acf_pacf(modelled_data, lags=50, color=colors[0])

    plot_preds_results(
        data=plotted_data,
        statsmodels_preds=statsmodels_preds,
        statsforecast_preds=statsforecast_preds,
    )

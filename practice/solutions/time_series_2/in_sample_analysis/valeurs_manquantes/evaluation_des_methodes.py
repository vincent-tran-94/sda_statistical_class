from typing import Tuple

import numpy as np
from pandas import Series, DataFrame
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from demos.utlis import get_time_series_with_missing_values
from practice.solutions.time_series_2.in_sample_analysis.valeurs_manquantes.valeurs_manquantes import (
    robust_moving_average_interpolation,
    polynomial_interpolation,
    machine_learning_interpolation,
    kalman_filter_interpolation,
)

IMPUTATION_METHODS: dict[str, callable] = {
    "Moyenne mobile": robust_moving_average_interpolation,
    "Interpolation polynomiale": polynomial_interpolation,
    "Machine learning": machine_learning_interpolation,
    "Filtre de Kalman": kalman_filter_interpolation,
}


def select_random_non_missing_chunck(data: Series, chunk_size: int) -> Tuple[int, int]:
    def select_chunk(data: Series, chunk_size: int) -> Tuple[int, int]:
        non_missing_indexes = data[~data.isnull()].index
        start_index = np.random.choice(non_missing_indexes, size=1)[0]
        end_index = start_index + chunk_size

        return start_index, end_index

    is_chunk_valid = False
    while not is_chunk_valid:
        start_index, end_index = select_chunk(data=data, chunk_size=chunk_size)
        is_chunk_valid = data[start_index:end_index].isna().sum() == 0

    return start_index, end_index


def run_imputing_methods_evaluation(
    data: Series,
    max_chunk_size: int,
    n_iterations: int = 10,
    methods_kwargs: dict[str, dict] = dict(),
):
    """
    Evaluates multiple imputation methods by randomly selecting non-missing chunks of varying sizes,
    masking them, and comparing imputation results to true values.

    Parameters:
    data (Series): The complete time series data with no missing values.
    max_chunk_size (int): The maximum chunk size to evaluate.
    n_iterations (int): Number of random chunks to evaluate per chunk size.
    methods_kwargs (dict[str, dict]): The method names as keys and their keyword arguments as values.

    Returns:
    DataFrame: DataFrame with mean squared errors, methods as rows, and chunk sizes as columns.
    """
    # Initialize an empty DataFrame to store MSE for each method and chunk size
    results_df = DataFrame(
        index=IMPUTATION_METHODS.keys(), columns=range(1, max_chunk_size + 1)
    )

    for chunk_size in range(1, max_chunk_size + 1):
        results = {method_name: [] for method_name in IMPUTATION_METHODS.keys()}

        for _ in range(n_iterations):
            # 1. Select a random chunk with no missing values
            chunk_start, chunk_end = select_random_non_missing_chunck(data, chunk_size)

            # 2. Copy the chunk and introduce missing values
            masked_chunk = data.copy()
            masked_chunk[chunk_start:chunk_end] = np.nan  # Mask all values in the chunk

            # 3. Apply each imputation method
            for method_name, method_func in IMPUTATION_METHODS.items():
                methods_kwargs = methods_kwargs.get(method_name, {})
                match method_name:
                    case "Machine learning":
                        imputed_data = method_func(
                            data=masked_chunk,
                            regressor=methods_kwargs.get("ml_regressor", SVR()),
                            differentiate_serie=methods_kwargs.get(
                                "differentiate_serie", False
                            ),
                        )
                    case "Moyenne mobile":
                        imputed_data = method_func(
                            data=masked_chunk,
                            window=methods_kwargs.get("window", 12),
                        )
                    case "Interpolation polynomiale":
                        imputed_data = method_func(
                            data=masked_chunk,
                            order=methods_kwargs.get("order", 3),
                        )
                    case "Filtre de Kalman":
                        imputed_data = method_func(
                            data=masked_chunk,
                            p=methods_kwargs.get("p", 3),
                            d=methods_kwargs.get("d", 1),
                            q=methods_kwargs.get("q", 1),
                            P=methods_kwargs.get("P", 1),
                            D=methods_kwargs.get("D", 1),
                            Q=methods_kwargs.get("Q", 1),
                            seasonal_frequency=methods_kwargs.get(
                                "seasonal_frequency", 12
                            ),
                        )
                    case _:
                        raise NotImplementedError(
                            f"Method {method_name} not implemented"
                        )

                # 4. Calculate the error (MSE) between the imputed values and the original values
                mse = mean_squared_error(
                    y_true=data[chunk_start:chunk_end],
                    y_pred=imputed_data[chunk_start:chunk_end],
                )
                results[method_name].append(mse)

        # Calculate the average MSE for each method for the current chunk size
        for method_name in IMPUTATION_METHODS.keys():
            results_df.at[method_name, chunk_size] = np.mean(results[method_name])

    return results_df


if __name__ == "__main__":
    data, _ = get_time_series_with_missing_values()
    evaluation_df = run_imputing_methods_evaluation(
        data=data, max_chunk_size=5, n_iterations=2
    )

    print("La méthode gagnante est :")
    print(evaluation_df.idxmin(axis=0))

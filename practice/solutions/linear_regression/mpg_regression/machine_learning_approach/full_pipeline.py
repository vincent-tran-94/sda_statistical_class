import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from demos.utlis import setup_plot
from practice.solutions.linear_regression.mpg_regression.machine_learning_approach.plotter import (
    plot_learning_curve,
    plot_results_with_mapie,
)
from practice.solutions.linear_regression.mpg_regression.machine_learning_approach.preprocessing import (
    create_preprocessor,
)


# Example usage within train_and_evaluate function
def train_and_evaluate(data: pd.DataFrame, target_column: str = "mpg"):
    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Full pipeline with model
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

    # Fit model
    full_pipeline.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = full_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(
        f"Root Mean Squared Error: {np.sqrt(mse):.2f} | Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}"
    )
    print(f"R^2 Score: {r2:.2f}")

    # Plot results with MAPIE jackknife intervals
    colors = setup_plot()
    plot_results_with_mapie(X_test, y_test, full_pipeline, colors=colors)
    plot_learning_curve(full_pipeline, X_train, y_train, colors=colors)


if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    data.drop(columns=["name"], inplace=True)
    train_and_evaluate(data=data)

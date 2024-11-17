import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from demos.utlis import setup_plot

if __name__ == "__main__":
    colors = setup_plot()

    # Step 1: Create some simulated data
    np.random.seed(42)
    n_samples = 500
    X = np.random.rand(n_samples, 1) * 10  # Random feature
    y = (
        2 * X.flatten() + np.sin(X.flatten()) * 10 + np.random.normal(0, 2, n_samples)
    )  # Nonlinear target with noise

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train Gradient Boosting Regressors for different quantiles
    quantiles = [0.1, 0.5, 0.9]  # 10th, 50th (median), and 90th quantiles
    models = {}
    predictions = {}

    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=100, max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)
        models[q] = model
        predictions[q] = model.predict(X_test)

    # Step 4: Predict and visualize the results
    X_test_sorted = np.sort(X_test, axis=0)  # Sort X_test for a cleaner plot
    y_test_sorted = y_test[np.argsort(X_test.flatten())]

    plt.figure(figsize=(12, 6))
    # Plot original test data
    plt.scatter(X_test, y_test, color="gray", alpha=0.6, label="Actual Data")

    # Plot the quantile predictions
    for q_idx, q in enumerate(quantiles):
        plt.plot(
            X_test_sorted,
            models[q].predict(X_test_sorted),
            label=f"{int(q*100)}th Quantile",
            linestyle="--",
            color=colors[q_idx],
        )

    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.title("Regression Quantile avec un Gradient Boosting")
    plt.grid(True)
    plt.show()
    # Step 5: Evaluate or print quantile predictions for a sample test instance
    sample_instance = np.array([[5]])  # Example instance for prediction
    sample_predictions = {q: models[q].predict(sample_instance)[0] for q in quantiles}
    print("Sample instance predictions (for X=5):", sample_predictions)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import glm

from demos.utlis import setup_plot

# Cas d'usage: modéliser la survie (qui est une variable continue) en fonction d'une variable explicative continue.
# Dans cet exemple, nous allons simuler des données de survie en utilisant un modèle de survie exponentiel.
# Nous allons ensuite ajuster un modèle de régression de Poisson pour modéliser la survie en fonction d'une variable continue.
# Enfin, nous allons comparer les courbes de survie de Kaplan-Meier avec les courbes de survie prédites par le modèle exponentiel.
# Les courbes de survie de Kaplan-Meier sont des estimations non paramétriques de la fonction de survie empirique.
# Les courbes de survie prédites par le modèle exponentiel sont basées sur la formule S(t) = exp(-λ * t), où λ est le taux de risque.

np.random.seed(42)
n = 100
covariate = np.random.normal(0, 1, n)
beta = 0.5
hazard_rate = np.exp(beta * covariate)
survival_time = np.random.exponential(1 / hazard_rate)
censoring_threshold = np.quantile(survival_time, 0.8)
event_occurred = (survival_time <= censoring_threshold).astype(int)
survival_time = np.minimum(survival_time, censoring_threshold)

# Create DataFrame
data = pd.DataFrame(
    {
        "survival_time": survival_time,
        "event_occurred": event_occurred,
        "covariate": covariate,
    }
)

# Step 2: Fit an Exponential GLM model
exponential_model = glm(
    formula="event_occurred ~ covariate",
    data=data,
    family=sm.families.Poisson(link=sm.families.links.log()),
)
result = exponential_model.fit()


# Step 3: Create a Kaplan-Meier survival curve manually
def kaplan_meier(data):
    """
    Calculate Kaplan-Meier survival function manually.
    """
    km_data = data.sort_values("survival_time")
    km_data["atrisk"] = np.arange(len(km_data), 0, -1)
    km_data["survival_prob"] = (
        km_data["atrisk"] - km_data["event_occurred"]
    ) / km_data["atrisk"]
    km_data["km_survival"] = km_data["survival_prob"].cumprod()
    return km_data[["survival_time", "km_survival"]]


# Separate data into two groups based on the median of the covariate
high_covariate = data["covariate"] > data["covariate"].median()
km_high = kaplan_meier(data[high_covariate])
km_low = kaplan_meier(data[~high_covariate])

colors = setup_plot()

# Step 4: Plot Kaplan-Meier Survival Curves
plt.figure(figsize=(10, 6))
plt.plot(
    km_high["survival_time"],
    km_high["km_survival"],
    label="Borne haute",
    color=colors[0],
)
plt.plot(
    km_low["survival_time"],
    km_low["km_survival"],
    label="Borne basse",
    color=colors[1],
)
plt.xlabel("Time")
plt.ylabel("Probabilité de survie")
plt.title("Courbe de Kaplan-Meier")
plt.legend()
plt.grid(True)

# Step 5: Plot the Exponential Model Predicted Survival Curve
# Predicted survival function: S(t) = exp(-λ * t), where λ is the hazard rate

# Calculate predicted survival curves based on the fitted model
mean_covariate_high = data.loc[high_covariate, "covariate"].mean()
mean_covariate_low = data.loc[~high_covariate, "covariate"].mean()

# Hazard rates for high and low covariate groups
lambda_high = np.exp(
    result.params["Intercept"] + result.params["covariate"] * mean_covariate_high
)
lambda_low = np.exp(
    result.params["Intercept"] + result.params["covariate"] * mean_covariate_low
)

# Time points for plotting the survival functions
time_points = np.linspace(0, data["survival_time"].max(), 100)

# Exponential survival functions
survival_high = np.exp(-lambda_high * time_points)
survival_low = np.exp(-lambda_low * time_points)

# Plot predicted survival curves
plt.plot(
    time_points,
    survival_high,
    color=colors[0],
    linestyle="--",
    label="Modèle exponentiel (Borne haute)",
)
plt.plot(
    time_points,
    survival_low,
    color=colors[1],
    linestyle="--",
    label="Modèle exponentiel (Borne basse)",
)

# Show the plot
plt.legend()
plt.show()

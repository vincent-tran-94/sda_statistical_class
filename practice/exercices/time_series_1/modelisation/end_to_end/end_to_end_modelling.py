from demos.utlis import get_time_series, setup_plot

if __name__ == "__main__":
    # todo : Quel est le meilleur modèle ? Faites une sélection de modèles en comparant les erreurs de prédiction
    # horizon de 36 points
    horizon = 36
    data = get_time_series()
    colors = setup_plot()

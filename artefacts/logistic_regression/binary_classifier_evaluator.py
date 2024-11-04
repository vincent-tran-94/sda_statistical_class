from numpy._typing import ArrayLike

from artefacts.logistic_regression.evaluation_functions import (
    print_binary_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_calibration_curve,
)


class BinaryClassifierEvaluator:
    """
    Classe permettant d'évaluer un modèle de classification binaire en affichant les métriques de classification.
    Les options d'affichage des métriques et des graphiques sont spécifiées dans le constructeur.
    """

    def __init__(
        self,
        print_metrics: bool = True,
        plot_confusion_matrix: bool = True,
        plot_roc_curve: bool = True,
        plot_calibration_curve: bool = True,
    ):
        self.print_metrics = print_metrics
        self.plot_confusion_matrix = plot_confusion_matrix
        self.plot_roc_curve = plot_roc_curve
        self.plot_calibration_curve = plot_calibration_curve

    def evaluate_classifier(
        self, y_true: ArrayLike, y_score: ArrayLike, colors: list[str]
    ) -> None:
        """
        Évalue un modèle de classification binaire en affichant les métriques de classification.

        Paramètres :
        ----------
        y_true : ArrayLike
            Les étiquettes réelles des classes.
        y_score: ArrayLike
            Les scores de probabilité prédits par le modèle (predict de statsmodels ou predict_proba de scikit-learn).
        """

        if self.print_metrics:
            print_binary_classification_metrics(y_true=y_true, y_pred=y_score > 0.5)
        if self.plot_confusion_matrix:
            plot_confusion_matrix(y_true=y_true, y_pred=y_score > 0.5, colors=colors)
        if self.plot_roc_curve:
            plot_roc_curve(y_true=y_true, y_score=y_score, colors=colors)
        if self.plot_calibration_curve:
            plot_calibration_curve(y_true=y_true, y_score=y_score, colors=colors)

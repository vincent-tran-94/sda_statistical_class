import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tools.typing import ArrayLike
from statsmodels.tsa.arima.model import ARIMA

from demos.utlis import get_time_series, setup_plot

import numpy as np
import pandas as pd
from typing import List
from numpy.typing import ArrayLike


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    # todo: Proposer des intervalles de prédiction en bootstrappant les erreurs
    # rappel de la méthode:
    # 1) Ajuster un modèle de prévision sur les données
    # 2) Récupérer les erreurs d'ajustement du modèle (.resid) dans le cas d'un modèle statsmodels
    # 3) Prévoir les h prochains points dans le futur
    # 4) Pour i allant de 1 à m (1000 par exemple):
    #   - Bootstrapper les erreurs de prévision et tirer h valeurs aléatoirement
    #   - Ajouter aux prévisions les h erreurs sélectionnés
    # 5) Récupérer les quantiles des prévisions bootstrappés correspondant au seuil de significativité voulu (0.05/0.95) ou (0.2/0.8) par exmple

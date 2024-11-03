import logging
from collections import OrderedDict
from typing import Dict, Any
from pandas import DataFrame

# Assuming PreprocessorInterface is defined elsewhere
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


class PreprocessingPipeline:
    def __init__(
        self,
        preprocessors: OrderedDict[str, PreprocessorInterface],
        preprocessor_kwargs: Dict[str, Dict[str, Any]] = None,
    ):
        """
        Initialize the preprocessing pipeline with preprocessors and optional kwargs.

        :param preprocessors: OrderedDict where keys are preprocessor names and values are preprocessor instances
        :param preprocessor_kwargs: Dictionary of kwargs for each preprocessor,
        with preprocessor names as keys and kwargs as dictionaries
        """
        self.preprocessors = preprocessors
        self.preprocessor_kwargs = preprocessor_kwargs or {}

    def fit(self, X: DataFrame) -> "PreprocessingPipeline":
        for preprocessor_name, preprocessor in self.preprocessors.items():
            kwargs = self.preprocessor_kwargs.get(preprocessor_name, {})
            logging.info(
                f"Fitting preprocessor {preprocessor_name} with kwargs {kwargs}"
            )
            self.preprocessors[preprocessor_name] = preprocessor.fit(X, **kwargs)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        for preprocessor_name, preprocessor in self.preprocessors.items():
            X = preprocessor.transform(X)
        return X

    def fit_transform(self, X: DataFrame) -> DataFrame:
        self.fit(X)
        return self.transform(X)

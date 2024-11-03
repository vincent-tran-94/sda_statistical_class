from abc import ABC, abstractmethod

from pandas import DataFrame


class PreprocessorInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: DataFrame, **kwargs) -> "PreprocessorInterface":
        pass

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X: DataFrame, **kwargs) -> DataFrame:
        pass

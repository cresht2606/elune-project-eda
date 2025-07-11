from abc import ABC, abstractmethod
import pandas as pd

class TransformerStep(ABC):
    """ Blueprint for all preprocessing addons"""

    @abstractmethod
    def fit(self, df : pd.DataFrame):
        """ Learn any parameters of further extension """
        return self

    @abstractmethod
    def transform(self, df : pd.DataFrame):
        """ Apply the actual transformation """
        ...

    @abstractmethod
    def fit_transform(self, df):
        """ Combination of previous functions """
        return self.fit(df).transform(df)


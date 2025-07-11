import pandas as pd
from typing import List
from abc import ABC, abstractmethod

#For unknown reasons, Kaggle didn't recognize the file transformer_step.py, so i decided to make a safe class copy
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
        
#Wrapper class that helps passing the categorical columns (Especially for Label & One Hot)
class ColumnTransformerStep:
    def __init__(self, transformer, cols : List[str]):
        self.cols = cols
        self.transformer = transformer

    def fit(self, df : pd.DataFrame):
        self.transformer.fit(df, self.cols)
        return self

    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        return self.transformer.transform(df, self.cols)

#Mainstream for preprocessing
class EDAPipeline:
    def __init__(self, steps : List[TransformerStep]):
        self.steps = steps

    def fit(self, df : pd.DataFrame):
        for step in self.steps:
            step.fit(df)
        return self

    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.transform(df)
        return df

    def fit_transform(self, df : pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)




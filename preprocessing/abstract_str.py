import pandas as pd
from abc import ABC, abstractmethod
from transformer_step import TransformerStep

#Abstract classes are implemented as blueprint for other outlier removers - encoders - transformers - scalers

class OutlierRemover(TransformerStep):
    @abstractmethod
    def fit(self, df : pd.DataFrame, cols : list[str] = None):
        #Optionally learn parameters, but OutlierRemover does not involve this step -> Skip
        return self
    @abstractmethod
    def transform(self, df : pd.DataFrame,cols : list[str] = None) -> pd.DataFrame:
        #Return cleaned dataframe, storing stripped rows in self.outliers
        ...

    def retrieve_outliers(self) -> pd.DataFrame:
        #Return the rows storing in self.outliers
        return getattr(self, "outliers", pd.DataFrame())
    

class EncodingStructure(TransformerStep):
    @abstractmethod
    def fit(self, df : pd.DataFrame):
        ...

    @abstractmethod
    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        ...

class TransformerStructure(TransformerStep):
    @abstractmethod
    def fit(self, df : pd.DataFrame):
        ...

    @abstractmethod
    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        ...

class ScalingStructure(TransformerStep):
    @abstractmethod
    def fit(self, df : pd.DataFrame):
        ...

    @abstractmethod
    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        ...



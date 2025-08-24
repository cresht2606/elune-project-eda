import pandas as pd
from typing import List

from transformer_step import TransformerStep

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

    def inverse_transform(self, df : pd.DataFrame) -> dict[str, pd.Series]:
        if hasattr(self.transformer, "inverse_transform"):
            return self.transformer.inverse_transform(df)
        else:
            raise NotImplementedError(f"{self.transformer.__class__.__name__} does not implement inverse_transform()")

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

    def inverse_transform(self, df : pd.DataFrame) -> pd.DataFrame:
        for step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                return step.inverse_transform(df)
        raise NotImplementedError("No step has inverse_transform() implemented.")






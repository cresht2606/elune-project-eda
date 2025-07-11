from abstract_str import ScalingStructure
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MinMax_Scaler(ScalingStructure):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.cols = []

    def fit(self, df : pd.DataFrame, cols : list[str]) -> None:
        self.cols = cols
        self.scaler.fit(df[cols])
        return self

    def transform(self, df : pd.DataFrame, cols : list[str]) -> pd.DataFrame:
        if self.scaler is None or not self.cols:
            raise RuntimeError("Incomplete Function: Expected to pass through fit(), then transform()")
        df = df.copy()
        cols = cols if cols is not None else self.cols
        df[cols] = self.scaler.transform(df[cols])
        return df

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df)
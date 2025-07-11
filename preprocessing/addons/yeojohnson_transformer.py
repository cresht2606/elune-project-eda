import pandas as pd
from abstract_str import TransformerStructure
from sklearn.preprocessing import PowerTransformer

#Yeo-Johnson does not require strict positive input, thereby skip the ValueError implementation

class YeoJohnsonTransformer(TransformerStructure):
    def __init__(self):
        self.transformer = None
        self.cols = []
    def fit(self, df : pd.DataFrame, cols : list[str]):
        self.cols = cols
        self.transformer = PowerTransformer(method = "yeo-johnson")
        self.transformer.fit(df[cols])
        return self

    def transform(self, df : pd.DataFrame, cols : list[str] = None) -> pd.DataFrame:
        if self.transformer is None or not self.cols:
            raise NotImplementedError("Incomplete Function: Expected to pass through fit(), then transform()")
        df = df.copy()
        cols = cols if cols is not None else self.cols
        df[cols] = self.transformer.transform(df[cols])
        return df

    def fit_transform(self, df, cols : list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df)
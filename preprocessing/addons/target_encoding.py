from abstract_str import EncodingStructure
import pandas as pd
from category_encoders import TargetEncoder

#TargetEncoding: Encodes each categorical columns with the mean of specific values
class TargetEncoding(EncodingStructure):
    def __init__(self, cols : list[str] , smoothing : float):
        self.cols = cols
        self.encoder = TargetEncoder(cols = cols, smoothing = smoothing)

    def fit(self, df : pd.DataFrame, y: pd.Series = None):
        if y is None:
            raise ValueError("Target Encoding requires a target 'y' Series")
        #Fit multiple categorical columns
        self.encoder.fit(df[self.cols].astype(str), y)
        return self

    def transform(self, df : pd.DataFrame, cols : list[str] = None) -> pd.DataFrame:
        if self.encoder is None or not self.cols:
            raise RuntimeError("Incomplete Function: Expected to pass through fit(), then transform()")
        #Transform will only touch 'self.cols' by default
        return pd.concat([df.drop(columns = self.cols), self.encoder.transform(df[self.cols].astype(str))], axis = 1)

    def fit_transform(self, df : pd.DataFrame, y : pd.Series = None) -> pd.DataFrame:
        return self.fit(df, y = y).transform(df)
from abstract_str import EncodingStructure
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

#TargetEncoding: Encodes each categorical columns with the mean of specific values
class TargetEncoding(EncodingStructure):
    def __init__(self, cols : list[str] , smoothing : float):
        self.cols = cols
        self.encoder = TargetEncoder(cols = cols, smoothing = smoothing)
        self.inverse_maps = {}

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

    #Mapping encoded values back to the closet original category
    def inverse_transform(self, df : pd.DataFrame) -> dict[str, pd.Series]:
        result = {}

        for col in self.cols:
            encoded_col = df[col]

            #Store the dict
            reverse_map = self.inverse_maps[col]

            mean_to_cat = {v: k for k, v in reverse_map.items()}
            means = np.array(list(mean_to_cat.keys()))

            #For each value, find the closet mean (approx inverse)
            def closest_category(val):
                idx = (np.abs(means - val)).argmin()
                return mean_to_cat[means[idx]]

            result[col] = encoded_col.apply(closest_category)

        return result

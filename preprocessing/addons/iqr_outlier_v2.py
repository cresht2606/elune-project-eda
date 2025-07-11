from abstract_str import OutlierRemover
import pandas as pd
from scipy.constants import value


class IQROutlierRemoverV2(OutlierRemover):
    def __init__(self, group_cols, value_cols):
        self.group_cols = group_cols
        #Passing either single or list of column names
        self.value_cols = value_cols if isinstance(value_cols, list) else [value_cols]
        self.outliers = pd.DataFrame()

    #Nothing done in computational method, so skip the fit process
    def fit(self, df : pd.DataFrame, cols : list[str] = None):
        return self

    def transform(self, df : pd.DataFrame, cols : list[str] = None) -> pd.DataFrame:
        vc = self.value_cols

        #Per group: q1, q3 & iqr
        q1 = df.groupby(self.group_cols)[vc].transform(lambda x : x.quantile(0.25))
        q3 = df.groupby(self.group_cols)[vc].transform(lambda x : x.quantile(0.75))
        iqr = q3 - q1

        #Bounds
        min_iqr = q1 - 1.5 * iqr
        max_iqr = q3 + 1.5 * iqr

        #Boolean mask for values within min and max_iqr
        mask = ((df[vc] >= min_iqr) & (df[vc] <= max_iqr)).all(axis = 1)

        #Store and return
        self.outliers = df.loc[~mask].reset_index(drop = True)
        return df.loc[mask].reset_index(drop = True)

    def fit_transform(self, df : pd.DataFrame, cols : list[str] = None):
        return self.fit(df, cols).transform(df, cols)
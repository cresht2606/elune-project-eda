from abstract_str import OutlierRemover
import pandas as pd
from scipy.constants import value


class ZScoreOutlierRemoverV2(OutlierRemover):
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

        #Calculate mean and standard deviation
        means = df.groupby(self.group_cols)[vc].transform("mean")
        stds = df.groupby(self.group_cols)[vc].transform("std")

        #Calculate z-score
        z = (df[vc] - means) / stds

        #Append z_{columns}
        for col in vc:
            df[f"z_{col}"] = z[col]

        #Mask and filter outliers
        mask = (z.abs() < 3).all(axis = 1)

        #Store and return
        self.outliers = df.loc[~mask].reset_index(drop = True)
        return df.loc[mask].reset_index(drop = True)

    def fit_transform(self, df : pd.DataFrame, cols : list[str] = None):
        return self.fit(df, cols).transform(df, cols)
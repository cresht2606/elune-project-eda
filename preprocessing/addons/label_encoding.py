from abstract_str import EncodingStructure
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class LabelEncoding(EncodingStructure):
    def __init__(self):
        self.encoder = {} #Label Encoding allows only one column per transformation
        self.cols = []

    def fit(self, df: pd.DataFrame, cols: list[str]) -> None:
        self.cols = cols
        for col in cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str)) #Strict string input for consistency
            self.encoder[col] = le
        return self

    def transform(self, df : pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if self.encoder is None or not self.cols:
            raise RuntimeError("Incomplete Function: Expected to pass through fit(), then transform()")
        df = df.copy()
        cols = cols if cols is not None else self.cols
        for col in cols:
            df[col] = self.encoder[col].transform(df[col].astype(str)) #Strict string input for consistency
        return df

    def fit_transform(self, df : pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df)

    #Inverse Transformation - Categorical Hashmap - Acting as a holder for Time Series export
    """
    From encoded feature names, use sklearn's One Hot encoder inverse transformation
    To revert the categorical columns (previously emerged),
    Store into the hashmap.
    """
    def inverse_transform(self, df : pd.DataFrame) -> dict[str, pd.Series]:
        return {
            col: {i: label for i, label in enumerate(self.encoder[col].classes_)}
            for col in self.cols
        }



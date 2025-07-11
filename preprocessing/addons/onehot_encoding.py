from abstract_str import EncodingStructure
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class OneHotEncoding(EncodingStructure):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.columns_out = []
        self.cols = []

    def fit(self, df: pd.DataFrame, cols: list[str]) -> None:
        self.cols = cols
        self.encoder.fit(df[cols])
        self.columns_out = self.encoder.get_feature_names_out(cols)
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if self.encoder is None or not self.cols:
            raise RuntimeError("Incomplete Function: Expected to pass through fit(), then transform()")
        df = df.copy()
        cols = cols if cols is not None else self.cols
        encoded = self.encoder.transform(df[cols])
        encoded_df = pd.DataFrame(encoded, columns=self.columns_out, index=df.index)
        return pd.concat([df.drop(columns=cols), encoded_df], axis=1)

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df)

    #Inverse Transformation - Categorical Hashmap - Acting as a holder for Time Series export
    """
    From encoded feature names, use sklearn's One Hot encoder inverse transformation
    To revert the categorical columns (previously emerged),
    Store into the hashmap.
    """
    def inverse_transform(self, df : pd.DataFrame) -> dict[str, pd.Series]:
        onehot_data = df[self.columns_out]
        original = self.encoder.inverse_transform(onehot_data)
        original_df = pd.DataFrame(original, columns = self.cols, index = df.index)
        return {
            col : original_df[col] for col in self.cols
        }

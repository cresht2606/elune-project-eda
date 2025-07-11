from abstract_str import OutlierRemover
import pandas as pd

class ZScoreOutlier(OutlierRemover):
    def __init__(self, group_cols, value_cols):
        self.group_cols = group_cols
        #Passing either single or list of column names
        self.value_cols = value_cols if isinstance(value_cols, list) else [value_cols]
        self.outliers = pd.DataFrame()

    #Nothing done in computational method, so skip the fit process
    def fit(self, df : pd.DataFrame, cols : list[str] = None):
        return self

    def transform(self, df : pd.DataFrame, cols : list[str] = None) -> pd.DataFrame:
        cleaned_parts = []
        outliers_parts = []
        for _, sub in df.groupby(self.group_cols):
            sub = sub.copy()
            if not all(pd.api.types.is_numeric_dtype(sub[c]) for c in self.value_cols):
                cleaned_parts.append(sub)
                continue
            mask = pd.Series(True, index=sub.index)
            for col in self.value_cols:
                z = (sub[col] - sub[col].mean()) / sub[col].std()
                prefix = f"z_{col}"
                sub[prefix] = z
                mask &= z.abs() < 3  # Combine mask across columns

        cleaned_parts.append(sub[mask])
        outliers_parts.append(sub[~mask])


        #Store outliers for later use (retrieve_outliers) and guard pd.concat to avoid calling on empty lists
        self.outliers = (
            pd.concat(outliers_parts, ignore_index=True)
            if outliers_parts else pd.DataFrame(columns=df.columns)
        )
        return (
            pd.concat(cleaned_parts, ignore_index=True)
            if cleaned_parts else pd.DataFrame(columns=df.columns)
        )

    def fit_transform(self, df : pd.DataFrame, cols : list[str] = None) -> pd.DataFrame:
        return self.fit(df, cols).transform(df, cols)
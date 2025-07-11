from abstract_str import OutlierRemover
import pandas as pd

class IQROutlierRemover(OutlierRemover):
    def __init__(self, group_cols, value_cols):
        self.group_cols = group_cols
        self.value_cols = value_cols if isinstance(value_cols, list) else [value_cols]
        self.outliers = pd.DataFrame()

    #Nothing done in computational method, so skip the fit process
    def fit(self, df : pd.DataFrame, cols : list[str] = None):
        return self

    def transform(self, df : pd.DataFrame, cols: list[str] = None) -> pd.DataFrame:
        cleaned_parts = []
        outliers_parts = []

        #Safe check on invalid or mistypoo columns
        missing = set(self.value_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        for _, sub in df.groupby(self.group_cols):
            sub = sub.copy()
            #Only function if strict numeric columns are valid
            if not all(pd.api.types.is_numeric_dtype(sub[c]) for c in self.value_cols):
                cleaned_parts.append(sub)
                continue
            else:
                q1 = sub[self.value_cols].quantile(0.25)
                q3 = sub[self.value_cols].quantile(0.75)
                iqr = q3 - q1
                max_iqr = q3 + 1.5 * iqr
                min_iqr = q1 - 1.5 * iqr

                mask = pd.Series(True, index = sub.index)
                for col in self.value_cols:
                    within_range = sub[col].between(min_iqr[col], max_iqr[col])
                    mask = mask & within_range

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

    def fit_transform(self, df : pd.DataFrame, cols: list[str] = None) -> pd.DataFrame:
        #Ensure return only cleaned dataframe 
        return self.fit(df, cols).transform(df, cols)

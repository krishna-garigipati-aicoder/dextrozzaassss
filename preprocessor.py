import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column=None, normalize=True, standardize=True):
        self.target_column = target_column
        self.normalize = normalize
        self.standardize = standardize
        self.imputers = {}
        self.label_encoders = {}
        self.scaler = None
        self.numeric_cols = None
        self.features = None

    def fit(self, X, y=None):
        df = X.copy()
        for col in df.columns:
            strategy = 'most_frequent' if df[col].dtype == 'object' else 'mean'
            imputer = SimpleImputer(strategy=strategy)
            df[col] = imputer.fit_transform(df[[col]]).flatten()
            self.imputers[col] = imputer

        for col in df.columns:
            if df[col].dtype == 'object' or df[col].apply(type).nunique() > 1:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        self.features = df.drop(columns=[self.target_column]) if self.target_column else df.copy()
        self.scaler = MinMaxScaler() if self.normalize else StandardScaler()
        df[self.features.columns] = self.scaler.fit_transform(self.features)

        self.fitted_df = df
        return self

    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            imputer = self.imputers.get(col)
            if imputer:
                df[col] = imputer.transform(df[[col]]).flatten()

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        features = df.drop(columns=[self.target_column]) if self.target_column and self.target_column in df.columns else df.copy()
        df[features.columns] = self.scaler.transform(features)
        return df

def preprocessing_pipeline(target_column=None, normalize=True, standardize=False):
    return CustomPreprocessor(target_column=target_column, normalize=normalize, standardize=standardize) 
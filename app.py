"""
DataJet: A One-Click AutoML Tabular Pipeline 
Author(s): Krishna, N.D.S. Nagadevi
Created: May 2025
"""

import streamlit as st
import pandas as pd
import os
from io import BytesIO
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import dill  

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

# === STREAMLIT APP ===

st.set_page_config(page_title="DataJet", page_icon="✈️", layout="wide")
st.title("✈️ DataJet: A tabular one-click AutoML pipeline")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if "csv_ready" not in st.session_state:
    st.session_state.csv_ready = False
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "csv_buffer" not in st.session_state:
    st.session_state.csv_buffer = None
if "model_buffer" not in st.session_state:
    st.session_state.model_buffer = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "val_score" not in st.session_state:
    st.session_state.val_score = None
if "eval_metric" not in st.session_state:
    st.session_state.eval_metric = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_cols = st.multiselect("Select Target Column(s)", options=columns)

    if st.button("Run Pipeline") and target_cols:
        import datetime


        try:
            with open("log_pipeline.txt", "a") as f:
                f.write(f"Pipeline run at {datetime.datetime.now()}\n")
        except Exception as e:
            st.warning(f"⚠️ Could not write to pipeline log: {e}")

        # Now run your actual pipeline
        # ...

        with st.spinner("Processing and training model..."):
            try:
                target_column = target_cols[0] if isinstance(target_cols, list) else target_cols

                prep_pipeline = preprocessing_pipeline(target_column=target_column, normalize=True, standardize=True)
                cleaned_df = prep_pipeline.fit_transform(df.copy())

                # Store cleaned CSV in memory
                csv_buffer = BytesIO()
                cleaned_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.session_state.csv_buffer = csv_buffer
                st.session_state.csv_ready = True

                # Check class balance for stratify
                target_counts = cleaned_df[target_column].value_counts()
                stratify = cleaned_df[target_column] if (target_counts >= 2).all() else None

                train_df, test_df = train_test_split(
                    cleaned_df,
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify
                )
                st.session_state.test_df = test_df

                model_path = "models/autogluon_model"
                predictor = TabularPredictor(label=target_column, path=model_path).fit(
                    train_df,
                    time_limit=100
                )

                st.session_state.eval_metric = predictor.eval_metric
                try:
                    best_model_name = predictor.get_model_best()
                    best_model_score = predictor.get_model_best_score()
                except Exception:
                    best_model_name = None
                    best_model_score = None

                st.session_state.best_model = best_model_name
                st.session_state.val_score = best_model_score

                combined_obj = (prep_pipeline, predictor)
                model_buffer = BytesIO()
                dill.dump(combined_obj, model_buffer)
                model_buffer.seek(0)
                st.session_state.model_buffer = model_buffer
                st.session_state.model_ready = True

                st.success("Pipeline complete! Model trained and saved.")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

if st.session_state.csv_ready:
    st.download_button(
        "📅 Download Cleaned CSV",
        data=st.session_state.csv_buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

import datetime

if st.session_state.model_ready:
    if st.download_button(
        label="📦 Download Model + Preprocessing Pipeline (.pkl)",
        data=st.session_state.model_buffer,
        file_name="autogluon_model_dill.pkl",
        mime="application/octet-stream"
    ):
        try:
            with open("log.txt", "a") as f:
                f.write(f"Download at {datetime.datetime.now()}\n")
        except Exception as e:
            st.warning(f"⚠️ Could not write to log: {e}")


if st.session_state.model_ready:
    st.subheader("📊 Best Model Evaluation")

    if st.session_state.best_model:
        st.write(f"**Best Model:** `{st.session_state.best_model}`")

    if st.session_state.val_score is not None and st.session_state.eval_metric is not None:
        st.write(f"**Validation {st.session_state.eval_metric.upper()}:** {st.session_state.val_score:.4f}")

    try:
        buffer = st.session_state.model_buffer
        buffer.seek(0)
        prep_pipeline, predictor = dill.load(buffer)

        test_df = st.session_state.test_df
        test_metrics = predictor.evaluate(test_df, auxiliary_metrics=True)

        metrics_df = pd.DataFrame.from_dict(test_metrics, orient='index', columns=['Score'])
        metrics_df = metrics_df.sort_index()

        st.write("**Full Test Set Evaluation:**")
        st.table(metrics_df)

    except Exception as e:
        st.error(f"Error during evaluation: {e}")

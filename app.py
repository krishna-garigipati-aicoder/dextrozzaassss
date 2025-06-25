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
# Import refactored modules
from preprocessor import CustomPreprocessor, preprocessing_pipeline
from autogluon_model import train_autogluon_model, save_model_and_preprocessor, evaluate_model, get_best_model_info

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
                st.session_state.target_column = target_column

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
                predictor = train_autogluon_model(train_df, target_column, model_path=model_path, time_limit=100)

                st.session_state.eval_metric = predictor.eval_metric
                try:
                    best_model_name, best_model_score = get_best_model_info(predictor)
                except Exception:
                    best_model_name = None
                    best_model_score = None

                st.session_state.best_model = best_model_name
                st.session_state.val_score = best_model_score

                model_buffer = save_model_and_preprocessor(prep_pipeline, predictor)
                st.session_state.model_buffer = model_buffer
                st.session_state.model_ready = True

                st.success("Pipeline complete! Model trained and saved.")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

if st.session_state.csv_ready and st.session_state.csv_buffer is not None:
    st.download_button(
        "📅 Download Cleaned CSV",
        data=st.session_state.csv_buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

import datetime

if st.session_state.model_ready and st.session_state.model_buffer is not None:
    st.subheader("🔮 Make Predictions")
    
    # Create tabs for different prediction methods
    pred_tab1, pred_tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction"])
    
    with pred_tab1:
        st.write("**Single Prediction** - Enter values for each feature:")
        
        try:
            buffer = st.session_state.model_buffer
            if buffer is not None:
                buffer.seek(0)
                prep_pipeline, predictor = dill.load(buffer)
                
                # Get feature columns (exclude target column)
                target_column = st.session_state.get('target_column', 'target')
                feature_columns = [col for col in df.columns if col != target_column]
                
                # Create input fields for each feature
                input_data = {}
                col1, col2 = st.columns(2)
                
                for i, col in enumerate(feature_columns):
                    with col1 if i % 2 == 0 else col2:
                        # Try to infer input type based on original data
                        sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                        
                        if pd.api.types.is_numeric_dtype(df[col]):
                            input_data[col] = st.number_input(f"{col}:", value=float(sample_value) if sample_value is not None else 0.0)
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            input_data[col] = st.date_input(f"{col}:", value=pd.to_datetime(sample_value).date() if sample_value is not None else pd.Timestamp.now().date())
                        else:
                            # For categorical/text columns, show unique values if not too many
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) <= 20:
                                input_data[col] = st.selectbox(f"{col}:", options=unique_vals.tolist(), index=0 if len(unique_vals) > 0 else None)
                            else:
                                input_data[col] = st.text_input(f"{col}:", value=str(sample_value) if sample_value is not None else "")
                
                if st.button("🔮 Predict", type="primary"):
                    try:
                        # Create DataFrame from input data
                        input_df = pd.DataFrame([input_data])
                        
                        # Apply preprocessing pipeline
                        processed_input = prep_pipeline.transform(input_df)
                        
                        # Make prediction
                        prediction = predictor.predict(processed_input)
                        prediction_proba = predictor.predict_proba(processed_input) if hasattr(predictor, 'predict_proba') and predictor.can_predict_proba else None
                        
                        # Decode prediction if it's a classification problem
                        decoded_prediction = prediction.iloc[0]
                        target_column = st.session_state.get('target_column')
                        if (target_column and 
                            hasattr(prep_pipeline, 'label_encoders') and 
                            target_column in prep_pipeline.label_encoders):
                            target_encoder = prep_pipeline.label_encoders[target_column]
                            try:
                                decoded_prediction = target_encoder.inverse_transform([prediction.iloc[0]])[0]
                            except:
                                # If inverse transform fails, keep original prediction
                                pass
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        st.write(f"**Prediction:** {decoded_prediction}")
                        
                        if prediction_proba is not None:
                            st.write("**Prediction Probabilities:**")
                            proba_df = pd.DataFrame(prediction_proba.iloc[0]).T
                            st.dataframe(proba_df)
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
            else:
                st.warning("Model buffer is not available for prediction.")
        except Exception as e:
            st.error(f"Error loading model for prediction: {e}")
    
    with pred_tab2:
        st.write("**Batch Prediction** - Upload CSV with your data to get predictions:")
        
        batch_file = st.file_uploader("Upload CSV for predictions", type=["csv"], key="batch_pred")
        
        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                st.write("**Preview of uploaded data:**")
                st.dataframe(batch_df.head())
                
                if st.button("🔮 Generate Predictions", type="primary"):
                    try:
                        buffer = st.session_state.model_buffer
                        if buffer is not None:
                            buffer.seek(0)
                            prep_pipeline, predictor = dill.load(buffer)
                            
                            # Apply preprocessing pipeline
                            processed_batch = prep_pipeline.transform(batch_df.copy())
                            
                            # Make predictions
                            predictions = predictor.predict(processed_batch)
                            predictions_proba = predictor.predict_proba(processed_batch) if hasattr(predictor, 'predict_proba') and predictor.can_predict_proba else None
                            
                            # Decode predictions if it's a classification problem
                            decoded_predictions = predictions.copy()
                            target_column = st.session_state.get('target_column')
                            if (target_column and 
                                hasattr(prep_pipeline, 'label_encoders') and 
                                target_column in prep_pipeline.label_encoders):
                                target_encoder = prep_pipeline.label_encoders[target_column]
                                try:
                                    decoded_predictions = pd.Series(target_encoder.inverse_transform(predictions), index=predictions.index)
                                except:
                                    # If inverse transform fails, keep original predictions
                                    pass
                            
                            # Create results DataFrame
                            results_df = batch_df.copy()
                            results_df['prediction'] = decoded_predictions
                            
                            if predictions_proba is not None:
                                # Add probability columns
                                proba_cols = predictions_proba.columns.tolist()
                                for col in proba_cols:
                                    results_df[f'prob_{col}'] = predictions_proba[col]
                            
                            st.success("✅ Batch predictions complete!")
                            st.write("**Preview of results:**")
                            st.dataframe(results_df.head())
                            
                            # Create download button for results
                            csv_buffer = BytesIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_buffer,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.warning("Model buffer is not available for prediction.")
                    except Exception as e:
                        st.error(f"❌ Batch prediction error: {e}")
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")

if st.session_state.model_ready and st.session_state.model_buffer is not None:
    st.subheader("📊 Best Model Evaluation")

    if st.session_state.best_model:
        st.write(f"**Best Model:** `{st.session_state.best_model}`")

    if st.session_state.val_score is not None and st.session_state.eval_metric is not None:
        st.write(f"**Validation {str(st.session_state.eval_metric).upper()}:** {st.session_state.val_score:.4f}")

    try:
        buffer = st.session_state.model_buffer
        if buffer is not None:
            buffer.seek(0)
            prep_pipeline, predictor = dill.load(buffer)

            test_df = st.session_state.test_df
            metrics_df = evaluate_model(predictor, test_df, auxiliary_metrics=True)

            st.write("**Full Test Set Evaluation:**")
            st.table(metrics_df)
        else:
            st.warning("Model buffer is not available for evaluation.")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

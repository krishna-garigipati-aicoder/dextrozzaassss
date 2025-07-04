from autogluon.tabular import TabularPredictor
import dill
from io import BytesIO
import pandas as pd

def train_autogluon_model(train_df, target_column, model_path="models/autogluon_model", time_limit = 50):
    predictor = TabularPredictor(label=target_column, path=model_path).fit(
        train_df,
        time_limit = time_limit
    )
    return predictor

def save_model_and_preprocessor(prep_pipeline, predictor):
    model_buffer = BytesIO()
    dill.dump((prep_pipeline, predictor), model_buffer)
    model_buffer.seek(0)
    return model_buffer

def evaluate_model(predictor, test_df, auxiliary_metrics=True):
    test_metrics = predictor.evaluate(test_df, auxiliary_metrics=auxiliary_metrics)
    metrics_df = pd.DataFrame.from_dict(test_metrics, orient='index')
    metrics_df = metrics_df.sort_index()
    return metrics_df

def get_best_model_info(predictor):
    leaderboard_df = predictor.leaderboard(silent=True)
    best_model_name = leaderboard_df.iloc[0]['model']
    best_model_score = leaderboard_df.iloc[0][predictor.eval_metric]
    return best_model_name, best_model_score 

from data_loading import load_and_merge
from data_preprocessing import preprocess
from feature_engineering import apply_feature_engineering
from model_training import train_models
from model_evaluation import evaluate_models, explain_models
from config import COMBINED_FILE, PREDICTION_FILE, SAMPLE_PREDICTION_FILE, SAMPLE_FILE
import pandas as pd

def main():
    try:
        print("Loading data from saved file...")
        df = pd.read_csv(COMBINED_FILE)
        print(f"Loaded {len(df)} rows from {COMBINED_FILE}")
    except FileNotFoundError:
        print("Saved file not found, processing from scratch...")
        df = preprocess()
        df = apply_feature_engineering(df)
        df.to_csv(COMBINED_FILE, index=False)
        df.head(50).to_csv(SAMPLE_FILE, index=False)

    model_action, model_success, model_zone, X_test, y_action_test, y_success_test, y_zone_test, le_action, df = train_models(df)

    df_with_predictions = evaluate_models(
        model_action, model_success, model_zone,
        X_test, X_test, X_test,
        y_action_test, y_success_test, y_zone_test,
        le_action, df
    )
    df_with_predictions.to_csv(PREDICTION_FILE, index=False)
    df_with_predictions.head(50).to_csv(SAMPLE_PREDICTION_FILE, index=False)

    explain_models(model_action, model_success, model_zone, X_test, X_test, X_test)

if __name__ == "__main__":
    main()

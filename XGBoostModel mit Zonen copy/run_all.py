from data_loading import load_and_merge
from data_preprocessing import preprocess
from feature_engineering import apply_feature_engineering
from model_training import train_models
from model_evaluation import evaluate_models, explain_models
from config import COMBINED_FILE, PREDICTION_FILE, SAMPLE_PREDICTION_FILE, SAMPLE_FILE
import pandas as pd
import warnings

# main execution function
def main():
    warnings.filterwarnings("ignore", message=".*NSSavePanel.*")

    # option 1: Load from saved file (for fast debugging only)
    try:
        print("Loading data from saved file.")
        df = pd.read_csv(COMBINED_FILE)
        print(f"Loaded {len(df)} rows from {COMBINED_FILE}")
    except FileNotFoundError:
        print("Saved file not found, processing from scratch.")
        df = preprocess()
        df = apply_feature_engineering(df)
        df.to_csv(COMBINED_FILE, index=False)
        print(f"Saved full dataset at {COMBINED_FILE}")
        df.head(50).to_csv(SAMPLE_FILE, index=False)
        print(f"Saved sample dataset for fast inspection at {SAMPLE_FILE}")

    """
    # option 2: Full processing
    load_and_merge()
    df = preprocess()
    df = apply_feature_engineering(df)
    df.to_csv(COMBINED_FILE, index=False)
    print(f"Saved full dataset at {COMBINED_FILE}")
    df.head(50).to_csv(SAMPLE_FILE, index=False)
    print(f"Saved sample dataset for fast inspection at {SAMPLE_FILE}")
    """

    # model training
    model_action, model_success, model_zone, X_test, y_action_test, y_success_test, y_zone_test, le_action, df = train_models(df)

    # model evaluation
    df_with_predictions = evaluate_models(
        model_action, model_success, model_zone,
        X_test, X_test, X_test,
        y_action_test, y_success_test, y_zone_test,
        le_action, df
    )
    df_with_predictions.to_csv(PREDICTION_FILE, index=False)
    print(f"Saved predictions at {PREDICTION_FILE}")
    df_with_predictions.head(50).to_csv(SAMPLE_PREDICTION_FILE, index=False)
    print(f"Saved sample predictions at {SAMPLE_PREDICTION_FILE}")

    # model explanation
    explain_models(model_action, model_success, model_zone, X_test, X_test, X_test)

if __name__ == "__main__":
    main()

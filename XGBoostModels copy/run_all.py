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

    """
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
    except Exception as e:
        print(f"Error loading saved file: {e}")  
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

    """
    # model training on test data
    #model_action, model_success, model_zone, X_test, y_action_test, y_success_test, y_zone_test, le_action, df = train_models(df)
    model_action, model_success, model_zone, X_action_test, X_success_test, X_zone_test, y_action_test, y_success_test, y_zone_test, le_action, df = train_models(df)

    # model evaluation on test data
    # model evaluation on test data
    df_with_predictions = evaluate_models(
        model_action, model_success, model_zone,
        X_action_test, X_success_test, X_zone_test, 
        y_action_test, y_success_test, y_zone_test,
        le_action, df
    )

    # make predictions on full dataset (test and train) and save
    # features for the action model
    features_action = model_action.get_booster().feature_names
    X_action_all = df_with_predictions[features_action].fillna(0)
    y_pred_action_all = model_action.predict(X_action_all)

    # features for the success model
    features_success = model_success.get_booster().feature_names
    X_success_all = df_with_predictions[features_success].fillna(0)
    y_pred_success_all = model_success.predict(X_success_all)

    # features for the zone model
    features_zone = model_zone.get_booster().feature_names
    X_zone_all = df_with_predictions[features_zone].fillna(0)
    y_pred_zone_all = model_zone.predict(X_zone_all)


    df_with_predictions["pred_next_action_cat_enc"] = y_pred_action_all
    df_with_predictions["pred_next_action_cat"] = le_action.inverse_transform(y_pred_action_all)
    df_with_predictions["pred_next_action_success"] = y_pred_success_all
    df_with_predictions["pred_next_zone"] = y_pred_zone_all

    # save predictions for all events
    df_with_predictions.to_csv(PREDICTION_FILE, index=False)
    print(f"Saved predictions for all events at {PREDICTION_FILE}")
    df_with_predictions.head(50).to_csv(SAMPLE_PREDICTION_FILE, index=False)
    print(f"Saved sample predictions at {SAMPLE_PREDICTION_FILE}")

    # model explanation
    explain_models(model_action, model_success, model_zone, X_action_test, X_success_test, X_zone_test)

if __name__ == "__main__":
    main()

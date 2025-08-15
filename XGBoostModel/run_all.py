from data_loading import load_and_merge
from data_preprocessing import load_and_preprocess
from feature_engineering import apply_feature_engineering
from model_training import train_model
from model_evaluation import evaluate_model
from config import COMBINED_FILE
from config import SAMPLE_FILE
import pandas as pd

def main():
    
    # Option 1: Load from saved file (fast)
    try:
        print("Loading data from saved file...")
        df = pd.read_csv(COMBINED_FILE)
        print(f"Loaded {len(df)} rows from {COMBINED_FILE}")
    except FileNotFoundError:
        print("Saved file not found, processing from scratch...")
        df = load_and_preprocess()
        df = apply_feature_engineering(df)
        df.to_csv(COMBINED_FILE, index=False)
        print(f"Saved full dataset at {COMBINED_FILE}")
        df.head(50).to_csv(SAMPLE_FILE, index=False)
        print(f"Saved sample dataset for fast inspection at {SAMPLE_FILE}")

    # Option 2: Full processing (uncomment if you want fresh data)
    #load_and_merge()
    # df = load_and_preprocess()
    # df = apply_feature_engineering(df)
    # df.to_csv(COMBINED_FILE, index=False)
    # print(f"Saved full dataset at {COMBINED_FILE}")
    # df.head(50).to_csv(SAMPLE_FILE, index=False)
    # print(f"Saved sample dataset for fast inspection at {SAMPLE_FILE}")

    model_action, model_success, X_model1_test, X_model2_test, y_model1_test, y_model2_test, le_action = train_model(df)
    evaluate_model(model_action, model_success, X_model1_test, X_model2_test, y_model1_test, y_model2_test, le_action)

if __name__ == "__main__":
    main()

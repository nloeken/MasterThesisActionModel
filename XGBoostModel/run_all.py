from data_loading import load_and_merge
from data_preprocessing import load_and_preprocess
from feature_engineering import apply_feature_engineering
from model_training import train_model
from model_evaluation import evaluate_model
from config import COMBINED_FILE
from config import SAMPLE_FILE

def main():
    #load_and_merge()
    df = load_and_preprocess()
    df = apply_feature_engineering(df)
    df.to_csv(COMBINED_FILE, index=False)
    print(f"Saved full dataset at {COMBINED_FILE}")
    df.to_csv(SAMPLE_FILE, index=False)
    print(f"Saved sample dataset for fast inspection at {SAMPLE_FILE}")

    model_action, model_success, X_test, y_action_test, y_success_test, le_action = train_model(df)
    evaluate_model(model_action, model_success, X_test, y_action_test, y_success_test, le_action)

if __name__ == "__main__":
    main()

from data_loading import load_and_merge
from data_preprocessing import load_and_preprocess
from feature_engineering import apply_feature_engineering
from model_training import train_model
from model_evaluation import evaluate_model
from config import COMBINED_FILE

def main():
    #load_and_merge()
    df = load_and_preprocess()
    df = apply_feature_engineering(df)
    df.to_csv(COMBINED_FILE, index=False)
    print(f"Saved full dataset at {COMBINED_FILE}")
    
    model, X_test, y_test, le_action = train_model(df)
    evaluate_model(model, X_test, y_test, le_action)

if __name__ == "__main__":
    main()

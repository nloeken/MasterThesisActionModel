import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, f1_score

def train_models(df):
    # --- Label Encoding ---
    le_action = LabelEncoder()
    df['next_action_cat_enc'] = le_action.fit_transform(df['next_action_cat'].astype(str))

    # --- One-hot Encoding ---
    df = pd.get_dummies(
        df,
        columns=['action_cat', 'prev_action_cat', 'team_name', 'position_name', 'match_phase', 'period'],
        prefix=['cur_act', 'prev_act', 'team', 'pos', 'phase', 'half']
    )

    # --- Feature Auswahl ---
    """
    #all features:
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'time_seconds', 'is_late_game', 'is_losing',
        'duration', 'possession_change', 'prev_event_success', 'combo_depth',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'))]
    
    """
    
    # new feature list for iteration 3
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'is_late_game', 'is_losing',
        'duration', 'prev_event_success',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'))]

    available_features = [f for f in base_features if f in df.columns]

    # --- Train/Test Split ---
    X = df[available_features].fillna(0)
    y_action = df['next_action_cat_enc']
    y_success = df['next_action_success']

    X_train, X_test, y_action_train, y_action_test, y_success_train, y_success_test = train_test_split(
        X, y_action, y_success, test_size=0.2, random_state=42, stratify=y_action
    )

    # --- Cross Validation Setup ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average="weighted")

    # --- Hyperparameter-Suchraum ---
    param_dist = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [200, 400, 800],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [1, 2, 5, 10]  # besonders für Imbalance wichtig
    }

    # --- Modell 1: Next Action Category ---
    model1 = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le_action.classes_),
        eval_metric="mlogloss",
    )

    search1 = RandomizedSearchCV(
        estimator=model1,
        param_distributions=param_dist,
        n_iter=20,  # Anzahl getesteter Kombinationen
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("\n=== Hyperparameter-Tuning: Model 1 (Next Action Category) ===")
    sample_weights_action = compute_sample_weight(class_weight="balanced", y=y_action_train)
    search1.fit(X_train, y_action_train, sample_weight=sample_weights_action)
    print("Beste Parameter (Model 1):", search1.best_params_)
    print("Bester CV-F1 (Model 1):", search1.best_score_)

    best_model1 = search1.best_estimator_

    # --- Modell 2: Success Probability ---
    model2 = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
    )

    search2 = RandomizedSearchCV(
        estimator=model2,
        param_distributions=param_dist,
        n_iter=20,
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("\n=== Hyperparameter-Tuning: Model 2 (Success Probability) ===")
    sample_weights_success = compute_sample_weight(class_weight="balanced", y=y_success_train)
    search2.fit(X_train, y_success_train, sample_weight=sample_weights_success)
    print("Beste Parameter (Model 2):", search2.best_params_)
    print("Bester CV-F1 (Model 2):", search2.best_score_)

    best_model2 = search2.best_estimator_

    return best_model1, best_model2, X_test, X_test, y_action_test, y_success_test, le_action, df

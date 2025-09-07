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
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'is_late_game', 'is_losing',
        'duration', 'prev_event_success',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'))]
    available_features = [f for f in base_features if f in df.columns]

    X = df[available_features].fillna(0)
    y_action = df['next_action_cat_enc']
    y_success = df['next_action_success']
    y_zone = df['next_action_zone']

    X_train, X_test, y_action_train, y_action_test, y_success_train, y_success_test, y_zone_train, y_zone_test = train_test_split(
        X, y_action, y_success, y_zone,
        test_size=0.2, random_state=42, stratify=y_action
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average="weighted")

    param_dist = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    def fit_xgb(search, X_train, y_train):
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_params = {
            "sample_weight": sample_weights,
            "eval_set": [(X_train, y_train)],
            "verbose": False
        }
        search.fit(X_train, y_train, **fit_params)
        return search.best_estimator_

    # --- Modell 1: Next Action Category ---
    model1 = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le_action.classes_),
        eval_metric="mlogloss"
    )
    search1 = RandomizedSearchCV(
        model1, param_distributions=param_dist, n_iter=5,
        scoring=f1_scorer, cv=cv, verbose=2, random_state=42, n_jobs=-1
    )
    print("\n=== Hyperparameter-Tuning: Model 1 (Next Action Category) ===")
    best_model1 = fit_xgb(search1, X_train, y_action_train)

    # --- Modell 2: Success Probability ---
    model2 = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss"
    )
    search2 = RandomizedSearchCV(
        model2, param_distributions=param_dist, n_iter=5,
        scoring=f1_scorer, cv=cv, verbose=2, random_state=42, n_jobs=-1
    )
    print("\n=== Hyperparameter-Tuning: Model 2 (Success Probability) ===")
    best_model2 = fit_xgb(search2, X_train, y_success_train)

    # --- Modell 3: Next Zone ---
    model3 = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=20,
        eval_metric="mlogloss"
    )
    search3 = RandomizedSearchCV(
        model3, param_distributions=param_dist, n_iter=5,
        scoring=f1_scorer, cv=cv, verbose=2, random_state=42, n_jobs=-1
    )
    print("\n=== Hyperparameter-Tuning: Model 3 (Next Zone) ===")
    best_model3 = fit_xgb(search3, X_train, y_zone_train)

    return (
        best_model1, best_model2, best_model3,
        X_test, y_action_test, y_success_test, y_zone_test,
        le_action, df
    )

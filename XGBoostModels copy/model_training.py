import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, f1_score

# function to train the three seperate models
def train_models(df):
    # label encoding
    le_action = LabelEncoder()
    df['next_action_cat_enc'] = le_action.fit_transform(df['next_action_cat'].astype(str))

    # one-hot encoding
    df = pd.get_dummies(
        df,
        columns=['action_cat', 'prev_action_cat', 'team_name', 'position_name', 'match_phase', 'period'],
        prefix=['cur_act', 'prev_act', 'team', 'pos', 'phase', 'half']
    )

    # feature selection
    """
    # include all features
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'time_seconds', 'is_late_game', 'is_losing',
        'duration', 'possession_change', 'prev_event_success', 'combo_depth', 'progress_to_goal',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'))]
    
    """
    # features included in final iteration
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'is_late_game', 'is_losing',
        'duration', 'possession_duration', 'prev_event_success',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'))]

    available_features = [f for f in base_features if f in df.columns]

    action_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'duration',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_'))]

    available_action_features = [f for f in action_features if f in df.columns]

    success_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'nearby_opponents',
        'orientation', 'free_teammates', 'duration', 'possession_duration', 'prev_event_success',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_', 'pos_', 'phase_'))]

    available_success_features = [f for f in success_features if f in df.columns]

    zone_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal',
        'nearby_opponents', 'orientation','free_teammates',
        'duration', 'possession_duration',
    ] + [col for col in df.columns if col.startswith(('cur_act_', 'prev_act_'))]

    available_zone_features = [f for f in zone_features if f in df.columns]

    all_needed_features = sorted(list(set(available_action_features + available_success_features + available_zone_features)))
    # train/test split
    #X = df[available_features].fillna(0)
    X = df[all_needed_features].fillna(0)
    y_action = df['next_action_cat_enc']
    y_success = df['next_action_success']
    y_zone = df['next_action_zone']


    # Einmaliger Split auf dem gesamten Datensatz
    X_train_full, X_test_full, y_action_train, y_action_test, y_success_train, y_success_test, y_zone_train, y_zone_test = train_test_split(
        X, y_action, y_success, y_zone,
        test_size=0.2, random_state=42, stratify=y_action
    )

    X_action_train = X_train_full[available_action_features]
    X_action_test = X_test_full[available_action_features]

    X_success_train = X_train_full[available_success_features]
    X_success_test = X_test_full[available_success_features]

    X_zone_train = X_train_full[available_zone_features]
    X_zone_test = X_test_full[available_zone_features]

    '''
    X_train, X_test, y_action_train, y_action_test, y_success_train, y_success_test, y_zone_train, y_zone_test = train_test_split(
        X, y_action, y_success, y_zone,
        test_size=0.2, random_state=42, stratify=y_action
    )
    '''

    # hyperparameter tuning with randomized search
    # cross validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # f1 scorer as metric
    f1_scorer = make_scorer(f1_score, average="weighted")

    # hyperparameter search space
    param_dist = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # class weights for action prediction
    classes = np.unique(y_action_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_action_train)
    class_weight_dict = dict(zip(classes, class_weights))
    weights_action = y_action_train.map(class_weight_dict)

    def fit_xgb(search, X_train, y_train, sample_weights=None):
        fit_params = {
            "eval_set": [(X_train, y_train)],
            "verbose": False
        }
        if sample_weights is not None:
            fit_params["sample_weight"] = sample_weights
        search.fit(X_train, y_train, **fit_params)
        return search.best_estimator_

    # model 1: next action category
    model1 = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le_action.classes_),
        eval_metric="mlogloss"
    )
    search1 = RandomizedSearchCV(
        estimator=model1,
        param_distributions=param_dist,
        n_iter=10,  
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("\n===== Hyperparameter-Tuning: Model 1 (Next Action Category) =====")
    #best_model1 = fit_xgb(search1, X_train, y_action_train, sample_weights=weights_action)
    best_model1 = fit_xgb(search1, X_action_train, y_action_train, sample_weights=weights_action)

    # model 2: success probability
    ratio = float(np.sum(y_success_train == 0)) / np.sum(y_success_train == 1)

    model2 = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=ratio
    )

    search2 = RandomizedSearchCV(
        estimator=model2,
        param_distributions=param_dist,
        n_iter=10,
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("\n===== Hyperparameter-Tuning: Model 2 (Success Probability) =====")
    #best_model2 = fit_xgb(search2, X_train, y_success_train)
    best_model2 = fit_xgb(search2, X_success_train, y_success_train)

    # model 3: next zone
    model3 = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=20,
        eval_metric="mlogloss"
    )
    search3 = RandomizedSearchCV(
        estimator=model3,
        param_distributions=param_dist,
        n_iter=10,
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\n===== Hyperparameter-Tuning: Model 3 (Next Zone) =====")
    #best_model3 = fit_xgb(search3, X_train, y_zone_train)
    best_model3 = fit_xgb(search3, X_zone_train, y_zone_train)

    # return trained models, test sets, label encoder and dataframe with predictions
    return (
        best_model1, best_model2, best_model3,
        X_action_test, X_success_test, X_zone_test, # Geänderte Rückgabewerte
        y_action_test, y_success_test, y_zone_test,
        le_action, df
    )

    """
     return (
        best_model1, best_model2, best_model3,
        X_test, y_action_test, y_success_test, y_zone_test,
        le_action, df
    )
    """
   

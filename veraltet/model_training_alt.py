import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def train_model(df):
    # label encoding
    le_action = LabelEncoder()
    df['action_cat_enc'] = le_action.fit_transform(df['action_cat'].astype(str))
    df['next_action_cat_enc'] = le_action.transform(df['next_action_cat'].astype(str))

    le_team = LabelEncoder()
    df['team_name_enc'] = le_team.fit_transform(df['team_name'].astype(str))
    
    le_position = LabelEncoder()
    df['position_name_enc'] = le_position.fit_transform(df['position_name'].astype(str))

    # one-hot encoding
    df = pd.get_dummies(df, columns=['match_phase', 'period'], prefix=['phase', 'half'])

    feature_cols = [
        'x', 'y', 'on_left_wing', 'on_right_wing', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_flank_zone', 'position_name_enc', 'duration',
        'time_seconds', 'is_late_game', 'is_losing', 'event_success', 'possession_change', 'action_cat_enc', 'team_name_enc',
        'prev_event_success', 'combo_depth', 'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation', 'free_teammates',
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('half_')]

    X = df[feature_cols]
    y_action = df['next_action_cat_enc']
    y_success = df['next_action_success']

    # Train/Test Split for both model 1 and model 2
    X_train, X_test, y_action_train, y_action_test, y_success_train, y_success_test = train_test_split(
        X, y_action, y_success, test_size=0.2, random_state=42
    )

    # Model 1: Predict next action category
    # compute sample weights for balancing classes
    sample_weights_action = compute_sample_weight(class_weight='balanced', y=y_action_train)
    model_action = xgb.XGBClassifier(objective='multi:softprob', num_class=len(le_action.classes_), eval_metric='mlogloss')
    model_action.fit(X_train, y_action_train, sample_weight=sample_weights_action)
    
    # Model 2: Success probability of the next action
    sample_weights_success = compute_sample_weight(class_weight='balanced', y=y_success_train)
    model_success = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model_success.fit(X_train, y_success_train, sample_weight=sample_weights_success)

    return model_action, model_success, X_test, y_action_test, y_success_test, le_action
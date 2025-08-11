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

    # one-hot encoding
    df = pd.get_dummies(df, columns=['match_phase', 'period'], prefix=['phase', 'half'])

    feature_cols = [
        'x', 'y', 'on_left_wing', 'on_right_wing', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_flank_zone', 'duration',
        'time_seconds', 'is_late_game', 'is_losing', 'event_success', 'possession_change', 'action_cat_enc', 'team_name_enc',
        'prev_event_success', 'combo_depth', 'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation', 'facing_goal', 'free_teammates',
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('half_')]

    X = df[feature_cols]
    y_action = df['next_action_cat_enc']
    y_success = df['next_action_success']

    # Train/Test Split
    X_train, X_test, y_train, y_test, y_success_train, y_success_test = train_test_split(
        X, y_action, y_success, test_size=0.2, random_state=42
    )

    # Compute sample weights for balancing classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(le_action.classes_), eval_metric='mlogloss')
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model, X_test, y_test, le_action

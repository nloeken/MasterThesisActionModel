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
    df['prev_action_cat_enc'] = df.groupby('match_id')['action_cat_enc'].shift(1)

    le_team = LabelEncoder()
    df['team_name_enc'] = le_team.fit_transform(df['team_name'].astype(str))
    
    le_position = LabelEncoder()
    df['position_name_enc'] = le_position.fit_transform(df['position_name'].astype(str))

    # one-hot encoding
    df = pd.get_dummies(df, columns=['match_phase', 'period'], prefix=['phase', 'half'])

    # all features 
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone', 
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation', 'free_teammates', 'position_name_enc',
        'time_seconds', 'is_late_game', 'is_losing', 'duration', 'possession_change', 'action_cat_enc',
        'team_name_enc', 'prev_action_cat_enc', 'prev_event_success', 'combo_depth', 
    ] + [col for col in df.columns if col.startswith('phase_')]

    # Filter out features that don't exist in the dataframe
    available_features = []
    for feature in base_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            print(f"Warning: Feature '{feature}' not found in dataframe")
    
    base_features = available_features

    # Model 1 features: base features + event_success (without duplicates)
    model1_features = base_features.copy()
    if 'event_success' not in model1_features and 'event_success' in df.columns:
        model1_features.append('event_success')

    # Model 2 features: base features (action_cat_enc already in base_features)
    model2_features = base_features.copy()
    
    print(f"Model 1 features: {len(model1_features)} features")
    print(f"Model 2 features: {len(model2_features)} features")
    
    # Features, target  and test/training-split for model 1 (next action category)
    X_model1 = df[model1_features].fillna(0)  # Fill NaN values with 0
    y_model1 = df['next_action_cat_enc']
    
    # Remove rows with NaN in target
    mask1 = ~y_model1.isna()
    X_model1 = X_model1[mask1]
    y_model1 = y_model1[mask1]
    
    X_model1_train, X_model1_test, y_model1_train, y_model1_test = train_test_split(
        X_model1, y_model1, test_size=0.2, random_state=42, stratify=y_model1
    )
    
    # Features, target  and test/training-split for model 2 (success probability of next action)
    X_model2 = df[model2_features].fillna(0)  # Fill NaN values with 0
    y_model2 = df['next_action_success']
    
    # Remove rows with NaN in target
    mask2 = ~y_model2.isna()
    X_model2 = X_model2[mask2]
    y_model2 = y_model2[mask2]
    
    X_model2_train, X_model2_test, y_model2_train, y_model2_test = train_test_split(
        X_model2, y_model2, test_size=0.2, random_state=42, stratify=y_model2
    )  

    # Model 1: Predict next action category
    # compute sample weights for balancing classes
    sample_weights_action = compute_sample_weight(class_weight='balanced', y=y_model1_train)
    model1 = xgb.XGBClassifier(objective='multi:softprob', num_class=len(le_action.classes_), eval_metric='mlogloss')
    model1.fit(X_model1_train, y_model1_train, sample_weight=sample_weights_action)
    
    # Model 2: Success probability of the next action
    sample_weights_success = compute_sample_weight(class_weight='balanced', y=y_model2_train)
    model2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model2.fit(X_model2_train, y_model2_train, sample_weight=sample_weights_success)

    return model1, model2, X_model1_test, X_model2_test, y_model1_test, y_model2_test, le_action, df
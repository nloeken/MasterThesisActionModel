import ast
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def extract_name(field):
    if isinstance(field, dict):
        return field.get("name")
    return field

# helper to get JSON-like strings from CSV
def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x

# helper to safe location data
def safe_location(loc):
    if isinstance(loc, list) and len(loc) == 2:
        return loc
    return [None, None]

# helper to estimate game phase
def get_match_phase(minute):
    if minute < 15: return "early_first"
    elif minute < 30: return "mid_first"
    elif minute < 45: return "late_first"
    elif minute < 60: return "early_second"
    elif minute < 75: return "mid_second"
    else: return "late_second"

# helper to get score status
def get_score_status(df, match_id, team_name, minute):
    # past events
    match_df = df[(df['match_id'] == match_id) & (df['minute'] <= minute)]

    # filter: shots with outcome 'Goal'
    goals = match_df[
        (match_df['type_name'] == 'Shot') &
        (match_df['shot'].apply(lambda x: isinstance(x, dict) and x.get('outcome', {}).get('name') == 'Goal'))
    ]

    team_goals = goals[goals['team_name'] == team_name].shape[0]
    opponent_goals = goals[goals['team_name'] != team_name].shape[0]

    if team_goals < opponent_goals:
        return 1  # behind
    elif team_goals == opponent_goals:
        return 0  # draw
    else:
        return -1  # lead

# helper to get movement angle
def get_movement_angle(row):
    start = row.get("location", [None, None])
    end = None

    if row["action_cat"] == "Dribble":
        carry = row.get("carry", {})
        if isinstance(carry, dict):
            end = carry.get("end_location", [None, None])
    elif row["action_cat"] == "Shot":
        shot = row.get("shot", {})
        if isinstance(shot, dict):
            end = shot.get("end_location", [None, None])
    elif row["action_cat"] == "Pass" or row["action_cat"] == "Cross":
        pas = row.get("pass", {})
        if isinstance(pas, dict):
            end = pas.get("end_location", [None, None])

    if (isinstance(start, list) and isinstance(end, list) and
        len(start) == 2 and len(end) == 2 and None not in start and None not in end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return np.arctan2(dy, dx)
    return np.nan

# helper to count opponents nearby
def count_opponents_nearby(row, radius=10):
    freeze = row.get("freeze_frame", [])
    if not isinstance(freeze, list):
        return np.nan
    loc = row.get("location", [None, None])
    if not (isinstance(loc, list) and None not in loc):
        return np.nan
    count = 0
    for player in freeze:
        if player.get("teammate") is False and isinstance(player.get("location"), list):
            dx = player["location"][0] - loc[0]
            dy = player["location"][1] - loc[1]
            if np.sqrt(dx**2 + dy**2) <= radius:
                count += 1
    return count

# helper to categorize events into action categories
def get_action_cat(row):
    if row["type_name"] in ["Carry", "Duel", "Dribble", "50/50"]:
        return "Dribble"
    if row["type_name"] == "Shot":
        return "Shot"
    if row["type_name"] in ["Half Start", "Clearance"]:
        return "Pass"
    if row["type_name"] == "Pass":
        # check for corner or cross
        pass_info = row.get("pass", {})
        if isinstance(pass_info, dict):
            if pass_info.get("type", {}).get("name", "") == "Corner":
                return "Cross"
            if pass_info.get("cross", False):
                return "Cross"
        return "Pass"
    return None

# helper for success estimation (for diffferent event types) 
def event_success(row):
    # Pass event types:
    if row["type_name"] == "Pass":
        outcome = row.get("pass", {}).get("outcome", {}).get("name", "")
        return int(outcome not in ["Incomplete", "Out"])
    if row["type_name"] == "Clearance":
        outcome = row.get("clearance", {}).get("outcome", {}).get("name", "")
        return 1
    if row["type_name"] == "Half Start":
        outcome = row.get("half start", {}).get("outcome", {}).get("name", "")
        return 1
    # Shot event types:
    if row["type_name"] == "Shot":
        outcome = row.get("shot", {}).get("outcome", {}).get("name", "")
        return int(outcome in ["Goal", "Post", "Saved", "Saved To Post"])
    # Dribble event types:
    if row["type_name"] == "50/50":
        outcome = row.get("50/50", {}).get("outcome", {}).get("name", "")
        return int(outcome in ["Won", "Success To Team"])
    if row["type_name"] == "Carry":
        outcome = row.get("carry", {}).get("outcome", {}).get("name", "")
        return 1
    if row["type_name"] == "Dribble":
        outcome = row.get("dribble", {}).get("outcome", {}).get("name", "")
        return int(outcome == "Complete")
    if row["type_name"] == "Duel":
        outcome = row.get("duel", {}).get("outcome", {}).get("name", "")
        return int(outcome in ["Won", "Success", "Success In Play", "Success Out"])
    return 0

# helper to count free teammates
def count_free_teammates(row, radius=5):
    freeze = row.get("freeze_frame", [])
    ball_location = row.get("location", [None, None])
    
    if not isinstance(freeze, list) or None in ball_location:
        return 0

    free_count = 0

    for player in freeze:
        if player.get("teammate") and isinstance(player.get("location"), list):
            teammate_loc = player["location"]
            is_marked = False

            for opponent in freeze:
                if not opponent.get("teammate") and isinstance(opponent.get("location"), list):
                    dx = opponent["location"][0] - teammate_loc[0]
                    dy = opponent["location"][1] - teammate_loc[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= radius:
                        is_marked = True
                        break

            if not is_marked:
                free_count += 1

    return free_count

# helper to plot SHAP values seperated by action classes
def plot_shap_classwise(model, X_test, le_action):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        for i, class_name in enumerate(le_action.classes_):
            shap.summary_plot(shap_values[i], X_test, feature_names=X_test.columns, show=True)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=True)


# helper to plot correlation matrix for features
def plot_feature_correlations(df, feature_cols):
    corr_matrix = df[feature_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.show()

# helper to plot XGBoost feature importance
def plot_xgb_importance(model, feature_names, max_features=20):
    xgb.plot_importance(model, max_num_features=max_features, importance_type='gain')
    plt.title("XGBoost Feature Importance")
    plt.show()






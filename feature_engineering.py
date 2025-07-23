import numpy as np
import pandas as pd
from utils import safe_location, get_match_phase, safe_eval

# success estimation (for diffferent event types) 
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

def apply_feature_engineering(df):
    # add team name as column
    df['team'] = df['team'].apply(safe_eval)
    df['team_name'] = df['team'].apply(lambda x: x.get('name') if isinstance(x, dict) else x if isinstance(x, str) else None)

    # extract action categories and event success
    df["action_cat"] = df.apply(get_action_cat, axis=1)
    df["event_success"] = df.apply(event_success, axis=1)

    # sort on time-dimension
    df = df.sort_values(by=['match_id', 'period', 'minute', 'second', 'index']).reset_index(drop=True)

    # create target variables
    df["next_action_cat"] = df.groupby("match_id")["action_cat"].shift(-1)
    df["next_action_success"] = df.groupby("match_id")["event_success"].shift(-1)

    # drop NANs in target columns 
    df = df.dropna(subset=["next_action_cat", "next_action_success"])

    # filter out events from penalty shootout (period 5)
    df = df[df['period'] != 5]

    # spatial features
    df['location'] = df['location'].apply(safe_eval).apply(safe_location)
    df[['x', 'y']] = pd.DataFrame(df['location'].tolist(), index=df.index)
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # coordinates of opponent's goal (always at the same position)
    goal_x, goal_y = 120, 40

    # features 1 and 2: distance and angle to opponent's goal
    # distance to goal (euclidean distance)
    df['distance_to_goal'] = np.sqrt((goal_x - df['x'])**2 + (goal_y - df['y'])**2).round(2)
    # angle to goal (in radians)
    df['angle_to_goal'] = np.arctan2(goal_y - df['y'], goal_x - df['x']).round(2)

    # feature 3: number of opponent players close to ball-carrier:
    df["nearby_opponents"] = df.apply(count_opponents_nearby, axis=1)
    # feature 4: orientation of ball-carrier 
    df["orientation"] = df.apply(get_movement_angle, axis=1)

    # time features
    df['time_seconds'] = df['minute'] * 60 + df['second']
    df['match_phase'] = df['minute'].apply(get_match_phase)

    # possession features
    # possession change
    df['possession_change'] = df['possession'].ne(df['possession'].shift())

    # contextual features
    df['prev_action_cat'] = df['action_cat'].shift(1)
    df['prev_event_success'] = df['event_success'].shift(1)

    return df

# function for feature 3
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

# function for feature 4
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

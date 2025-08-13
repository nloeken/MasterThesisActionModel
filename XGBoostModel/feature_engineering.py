import numpy as np
import pandas as pd
from utils import safe_location, get_match_phase, safe_eval, get_score_status, get_movement_angle, count_opponents_nearby, get_action_cat, event_success, count_free_teammates, get_last_n_events, get_time_since_last_event, get_progress_to_goal, get_action_patterns
from config import MAX_EVENT_GAP

def apply_feature_engineering(df):
    # add team name as column
    df['team'] = df['team'].apply(safe_eval)
    df['team_name'] = df['team'].apply(lambda x: x.get('name') if isinstance(x, dict) else x if isinstance(x, str) else None)

    # add position name as column
    df['position'] = df['position'].apply(safe_eval)
    df['position_name'] = df['position'].apply(lambda x: x.get('name') if isinstance(x, dict) else x if isinstance(x, str) else None)

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
    # on left wing (boolean, 1 if on left wing, 0 otherwise)
    df['on_left_wing'] = ((df['x'] >= 90) & (df['y'] <= 18)).astype(int)
    # on right wing (boolean, 1 if on right wing, 0 otherwise)
    df['on_right_wing'] = ((df['x'] >= 90) & (df['y'] >= 62)).astype(int)

    # coordinates of opponent's goal (always at the same position)
    goal_x, goal_y = 120, 40

    # feature 1: distance to goal (euclidean distance)
    df['distance_to_goal'] = np.sqrt((goal_x - df['x'])**2 + (goal_y - df['y'])**2).round(2)
    # feature 2: angle to goal (in radians)
    df['angle_to_goal'] = np.arctan2(goal_y - df['y'], goal_x - df['x']).round(2)
    # feature 3: in box (boolean, 1 if in box, 0 otherwise)
    df['in_box'] = ((df['x'] > 103.5) & (df['y'] > 20) & (df['y'] < 60)).astype(int)
    # feature 4: in flank zone (boolean, 1 if in flank zone, 0 otherwise)
    df['in_flank_zone'] = ((df['on_left_wing'] == 1) | (df['on_right_wing'] == 1)).astype(int)

    # feature 5: number of opponent players close to ball-carrier:
    df["nearby_opponents"] = df.apply(count_opponents_nearby, axis=1)
    df['high_pressure'] = (df['nearby_opponents'] >= 3).astype(int)
    df['low_pressure'] = (df['nearby_opponents'] == 0).astype(int)
    # feature 6: orientation of ball-carrier
    df["orientation"] = df.apply(get_movement_angle, axis=1)
    df['facing_goal'] = (df['orientation'] > 0.75).astype(int)  # Beispiel-Threshold, ggf. anpassen
    # free teammates in frame
    df['free_teammates'] = df.apply(count_free_teammates, axis=1)


    # time features
    df['time_seconds'] = df['minute'] * 60 + df['second']
    df['match_phase'] = df['minute'].apply(get_match_phase)
    # late game (boolean, 1 if minute >= 80, 0 otherwise)
    df['is_late_game'] = (df['minute'] >= 80).astype(int)
     # score at minute
    df['is_losing'] = df[['match_id', 'team_name', 'minute']].apply(
        lambda row: get_score_status(df, row['match_id'], row['team_name'], row['minute']) == 1,
        axis=1
    ).astype(int)

    # possession features
    # possession change
    df['possession_change'] = df['possession'].ne(df['possession'].shift())

    # contextual features
    df['prev_action_cat'] = df['action_cat'].shift(1)
    df['prev_event_success'] = df['event_success'].shift(1)
    df['combo_depth'] = df.groupby('possession').cumcount() 
    df = get_last_n_events(df, n=3)
    df = get_time_since_last_event(df)
    # filter out sequences with a long time since last event
    df = df[df['time_since_last_event'] <= MAX_EVENT_GAP]
    df = get_progress_to_goal(df)
    df = get_action_patterns(df, n=3)

    return df




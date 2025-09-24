import numpy as np
import pandas as pd
from utils import safe_location, get_match_phase, safe_eval, get_score_status, get_movement_angle, count_opponents_nearby, get_action_cat, event_success, count_free_teammates, get_time_since_last_event, get_progress_to_goal

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
    # basic locations
    df['location'] = df['location'].apply(safe_eval).apply(safe_location)
    df[['x', 'y']] = pd.DataFrame(df['location'].tolist(), index=df.index)
    # feature 1: x coordinate
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    # feature 2: y coordinate
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # positions relative to the field
    # coordinates of opponent's goal (always at the same position)
    goal_x, goal_y = 120, 40
    # feature 3: distance to goal (euclidean distance)
    df['distance_to_goal'] = np.sqrt((goal_x - df['x'])**2 + (goal_y - df['y'])**2).round(2)
    # feature 4: angle to goal (in radians)
    df['angle_to_goal'] = np.arctan2(goal_y - df['y'], goal_x - df['x']).round(2)
    # feature 5: in penalty area (boolean, 1 if in box, 0 otherwise)
    df['in_box'] = ((df['x'] > 103.5) & (df['y'] > 20) & (df['y'] < 60)).astype(int)

    # on left wing (boolean, 1 if on left wing, 0 otherwise)
    df['on_left_wing'] = ((df['x'] >= 90) & (df['y'] <= 18)).astype(int)
    # on right wing (boolean, 1 if on right wing, 0 otherwise)
    df['on_right_wing'] = ((df['x'] >= 90) & (df['y'] >= 62)).astype(int)
    # feature 6: in cross zone (boolean, 1 if in cross zone, 0 otherwise)
    df['in_cross_zone'] = ((df['on_left_wing'] == 1) | (df['on_right_wing'] == 1)).astype(int)

    # feature 7: number of opponent players close to ball-carrier:
    df["nearby_opponents"] = df.apply(count_opponents_nearby, axis=1)
    # feature 8: high pressure (boolean, 1 if >= 3 opponents nearby, 0 otherwise)
    df['high_pressure'] = (df['nearby_opponents'] >= 3).astype(int)
    # feature 9: low pressure (boolean, 1 if no opponents nearby, 0 otherwise)
    df['low_pressure'] = (df['nearby_opponents'] == 0).astype(int)
    # feature 10: orientation of ball-carrier
    df["orientation"] = df.apply(get_movement_angle, axis=1)
    #TODO: number of free teammates in frame (aktuell), Kegel nutzen um number of free teammates im Sichtfeld zu berechnen
    # feature 11: free teammates in frame
    df['free_teammates'] = df.apply(count_free_teammates, axis=1)

    # time features
    # feature 12: time in seconds
    df['time_seconds'] = df['minute'] * 60 + df['second']
    # feature 13: match phase
    df['match_phase'] = df['minute'].apply(get_match_phase)
    # feature 14: is late game (boolean, 1 if minute >= 80, 0 otherwise)
    df['is_late_game'] = (df['minute'] >= 80).astype(int)
    # feature 15: is losing (boolean, 1 if team is losing at minute, 0 otherwise)
    df['is_losing'] = df[['match_id', 'team_name', 'minute']].apply(
        lambda row: get_score_status(df, row['match_id'], row['team_name'], row['minute']) == 1,
        axis=1
    ).astype(int)

    # contextual features
    # feature 16: previous action category
    df['prev_action_cat'] = df['action_cat'].shift(1)
    # feature 17: previous event success
    df['prev_event_success'] = df['event_success'].shift(1)
    # feature 18: possession change (boolean, 1 if possession changed, 0 otherwise)   
    df['possession_change'] = df['possession'].ne(df['possession'].shift())
    # feature 19: combo depth (number of events in possession)
    df['combo_depth'] = df.groupby('possession').cumcount()
    # feature 20: duration of event
    df['duration'] = df['duration'].round(2)
    # feature 21: time since last event
    df = get_time_since_last_event(df)
    # feature 22: progress to goal (distance to goal relative to distance at start of possession)
    df = get_progress_to_goal(df)

    return df




import os
import pandas as pd
from tqdm import tqdm
from config import EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR, MAIN_EVENT_TYPES, META_DIR
from utils import extract_name

# function to sequentially load and merge event and positional data
def load_and_merge():
    
    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith('.json')]
    match_ids = [os.path.splitext(f)[0] for f in event_files]
    meta_info = []

    for match_id in tqdm(match_ids, desc="Processing matches"):
        try:
            events_path = os.path.join(EVENTS_DIR, f'{match_id}.json')
            positional_path = os.path.join(THREE_SIXTY_DIR, f'{match_id}.json')

            # skip matches only having event data
            if not os.path.exists(positional_path):
                continue

            events_df = pd.read_json(events_path)
            positional_df = pd.read_json(positional_path)

            # For exploratory purposes, extract team names from Starting XI event
            starting_xi = events_df[events_df['type'].apply(lambda x: x['name']) == 'Starting XI'].reset_index(drop=True)
            if len(starting_xi) >= 2:
                team_1_name = starting_xi.loc[0, 'team']['name']
                team_2_name = starting_xi.loc[1, 'team']['name']
            else:
                team_1_name, team_2_name = "unknown", "unknown"

            meta_info.append({
                "match_id": match_id,
                "team_1": team_1_name,
                "team_2": team_2_name
            })

            # merge the event and positional df
            df = pd.merge(
                events_df,
                positional_df,
                left_on='id',
                right_on='event_uuid',
                how='inner'
            )

            # extract and store names separately
            df["type_name"] = df["type"].apply(extract_name)
            df["player_name"] = df["player"].apply(extract_name)
            df["team_name"] = df["possession_team"].apply(extract_name)
            df["position_name"] = df["position"].apply(extract_name)

            # add column match_id
            df["match_id"] = match_id

            # filter: only keep events from main event types
            df = df[df['type_name'].isin(MAIN_EVENT_TYPES)].reset_index(drop=True)

            # filter: remove events from penalty shootout (period 5)
            df = df[df['period'] != 5]

            # only keep needed columns
            keep_cols = [
                "id", "index", "period", "minute", "second", "duration",
                "type", "type_name", "team", "team_name", "position", "position_name",
                "possession", "possession_team", "player", "player_name", "location",
                "pass", "carry", "dribble", "shot", "duel", "clearance", "freeze_frame",
                "match_id"
            ]
            df = df[[col for col in keep_cols if col in df.columns]]

            # sort time-wise
            df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

            # save merged df as csv
            out_path = os.path.join(MERGED_DIR, f"contextualevents_{match_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"[{match_id}] saved at {out_path}")

        except Exception as e:
            print(f"[{match_id}] Error: {e}")

    # save meta info as csv
    if meta_info:
        meta_df = pd.DataFrame(meta_info)
        os.makedirs(META_DIR, exist_ok=True)
        meta_out = os.path.join(META_DIR, "matches_overview.csv")
        meta_df.to_csv(meta_out, index=False)
        print(f"Übersicht gespeichert unter: {meta_out}")

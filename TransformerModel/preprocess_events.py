import pandas as pd
import numpy as np

# Aktion-Mapping
ACT_GROUPS = {
    'Pass': ['Pass', 'Smart pass'],
    'Shot': ['Shot', 'Goal'],
    'Dribble': ['Dribble'],
    'Cross': ['Cross'],
    'Possession End': ['Possession lost', 'Foul', 'Interception']
}

def map_action(action):
    for k, v in ACT_GROUPS.items():
        if action in v:
            return k
    return 'Other'

def preprocess_events(input_csv='merged_events.csv', output_csv='preprocessed_events.csv'):
    df = pd.read_csv(input_csv)
    
    # Filter irrelevant Events
    df = df[~df['type_name'].isin(['Own Goal', 'Start XI', 'Ball Receipt*', 'Pressure', 'Foul'])]
    
    # Map Actions
    df['act'] = df['type_name'].apply(map_action)
    
    # DeltaT berechnen
    df['match_seconds'] = df['minute']*60 + df['second']
    df = df.sort_values(['match_id', 'match_seconds'])
    df['deltaT'] = df.groupby('match_id')['match_seconds'].diff().fillna(0)
    
    # Pitch Features (Zonen)
    df['zone_x'] = (df['location_x'] // 10).astype(int)
    df['zone_y'] = (df['location_y'] // 10).astype(int)
    
    # Optional: Tor Distanz/Winkel
    df['dist_to_goal'] = np.sqrt((100 - df['location_x'])**2 + (50 - df['location_y'])**2)
    
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed events saved to {output_csv}")
    return df

def main():
    preprocess_events()

if __name__ == "__main__":
    main()

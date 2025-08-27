import json
import pandas as pd
import os
import glob
from config import EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR

def json_files_to_df(folder_path: str) -> pd.DataFrame:
    all_data = []
    for file_path in glob.glob(os.path.join(folder_path, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def merge_event_360(events_dir: str, frames_dir: str, output_file: str):
    event_df = json_files_to_df(events_dir)
    frames_df = json_files_to_df(frames_dir)
    
    # Merge auf event_uuid bzw. id
    merged_df = event_df.merge(frames_df, left_on='id', right_on='event_uuid', how='left')
    
    # CSV speichern
    merged_df.to_csv(output_file, index=False)
    print(f"Merged events saved to {output_file}")
    return merged_df

def main():
    merged_df = merge_event_360(EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR)
    print(merged_df.head())

if __name__ == "__main__":
    main()

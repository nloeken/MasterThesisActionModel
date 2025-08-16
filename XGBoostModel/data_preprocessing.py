import os
import pandas as pd
from tqdm import tqdm
from config import MERGED_DIR
from utils import safe_eval

def preprocess():
    all_files = [os.path.join(MERGED_DIR, f) for f in os.listdir(MERGED_DIR) if f.endswith('.csv')]
    dfs = []

    for file in tqdm(all_files, desc="Loading match files"):
        df = pd.read_csv(file)
        # use helper function safe_eval to get JSON-like strings from certain csv columns (for better handling)
        for col in ["type", "pass", "clearance", "half start", "shot", "50/50", "carry", "dribble", "duel", "freeze_frame"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval)
        dfs.append(df)

    # concat all dfs
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

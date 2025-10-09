import pandas as pd
import numpy as np
from socceraction.xthreat import ExpectedThreat
from socceraction.xthreat import load_model
from config import PREDICTION_FILE, XT_MATRIX

# settings of playing pitch (StatsBomb: 120x80)
PITCH_LENGTH = 120
PITCH_WIDTH = 80
N_X, N_Y = 12, 8  # grid size for xT model

# function to map (x,y) to zone in 12x8 grid
def get_zone_from_xy(x, y):
    if pd.isna(x) or pd.isna(y):
        return None
    # Ensure coordinates are within pitch boundaries
    x = max(0, min(x, PITCH_LENGTH))
    y = max(0, min(y, PITCH_WIDTH))
    col = int(min(x / (PITCH_LENGTH / N_X), N_X - 1))
    row = int(min(y / (PITCH_WIDTH / N_Y), N_Y - 1))
    return row * N_X + col

# function to evaluate xT values and rankings
def evaluate_xt_comparison(prediction_file=PREDICTION_FILE):
    # load prediction data
    df = pd.read_csv(prediction_file)
    print(f"Loaded {len(df)} rows from {prediction_file}")

    # load xT-matrix
    xt_model = load_model(XT_MATRIX)

    # compute current zones if not already present
    if "zone" not in df.columns:
        df["zone"] = df.apply(lambda r: get_zone_from_xy(r["x"], r["y"]), axis=1)

    # --- NEU: 'actual_next_zone' und 'actual_action_success' ERSTELLEN ---
    # We group by match and team to handle sequences of possession correctly.
    # .shift(-1) gets the value from the *next* row within each group.
    df['actual_next_zone'] = df.groupby(['match_id', 'team'])['zone'].shift(-1)

    # The actual action was 'successful' if it led to a next action by the same team.
    # If 'actual_next_zone' is NaN, it's the end of a sequence (e.g., shot, lost possession).
    df['actual_action_success'] = np.where(df['actual_next_zone'].notna(), 1, 0)
    # --------------------------------------------------------------------

    # current xT value (is the same for both scenarios)
    df["xt_current"] = df["zone"].apply(
        lambda z: xt_model.xT[int(z) // N_X, int(z) % N_X] if pd.notna(z) else 0
    )

    # --- 1. PREDICTED xT CALCULATION ---
    df["xt_predicted_next"] = df["pred_next_zone"].apply(
        lambda z: xt_model.xT[int(z) // N_X, int(z) % N_X] if pd.notna(z) else 0
    )
    df["delta_xt_predicted"] = (df["xt_predicted_next"] - df["xt_current"]) * df["pred_next_action_success"]

    # --- 2. ACTUAL xT CALCULATION ---
    df["xt_actual_next"] = df["actual_next_zone"].apply(
        lambda z: xt_model.xT[int(z) // N_X, int(z) % N_X] if pd.notna(z) else 0
    )
    df["delta_xt_actual"] = (df["xt_actual_next"] - df["xt_current"]) * df["actual_action_success"]


    # --- RANKINGS BASED ON PREDICTIONS ---
    player_ranking_pred = (
        df.groupby("player_name")["delta_xt_predicted"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"delta_xt_predicted": "total_xt_predicted"})
    )
    
    # --- RANKINGS BASED ON ACTUAL ACTIONS ---
    player_ranking_actual = (
        df.groupby("player_name")["delta_xt_actual"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"delta_xt_actual": "total_xt_actual"})
    )

    # Merge rankings for direct comparison
    player_comparison = pd.merge(player_ranking_pred, player_ranking_actual, on="player_name", how="outer").fillna(0)

    return df, player_comparison


# main execution function
if __name__ == "__main__":
    df_xt, player_comparison = evaluate_xt_comparison()

    print("\n===== Top Players Comparison (Predicted vs. Actual xT) =====")
    print(player_comparison.head(20))
    
    # You can sort to find discrepancies
    print("\n===== Players UNDERestimated by the model (Actual >> Predicted) =====")
    player_comparison['diff'] = player_comparison['total_xt_actual'] - player_comparison['total_xt_predicted']
    print(player_comparison.sort_values('diff', ascending=False).head(10))

    print("\n===== Players OVERestimated by the model (Predicted >> Actual) =====")
    print(player_comparison.sort_values('diff', ascending=True).head(10))
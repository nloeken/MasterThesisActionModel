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
    col = int(min(x / (PITCH_LENGTH / N_X), N_X - 1))
    row = int(min(y / (PITCH_WIDTH / N_Y), N_Y - 1))
    return row * N_X + col

# function to evaluate xT values and rankings
def evaluate_xt(prediction_file=PREDICTION_FILE):
    # load prediction data
    df = pd.read_csv(prediction_file)
    print(f"Loaded {len(df)} rows from {prediction_file}")
    print(df.columns)
    print(df[["x", "y", "zone"]].head(10))

    # load xT-matrix (original values from Karun Singh)
    xt_model = load_model(XT_MATRIX)

    # compute zones if not already present
    if "zone" not in df.columns:
        df["zone"] = df.apply(lambda r: get_zone_from_xy(r["x"], r["y"]), axis=1)

    # current & predicted xT values
    df["xt_current"] = df["zone"].apply(
        lambda z: xt_model.xT[int(z) // N_X, int(z) % N_X] if pd.notna(z) else 0
    )
    df["xt_next"] = df["pred_next_zone"].apply(
        lambda z: xt_model.xT[int(z) // N_X, int(z) % N_X] if pd.notna(z) else 0
    )

    # delta xT = change in xT weighted by success probability of next action
    df["delta_xT"] = (df["xt_next"] - df["xt_current"]) * df["pred_next_action_success"]

    # player ranking
    player_ranking = (
        df.groupby("player_name")["delta_xT"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"delta_xT": "total_xT"})
    )

    player_ranking_per_game = (
    df.groupby(["match_id", "player_name"])["delta_xT"]
    .sum()
    .reset_index()
    .sort_values(["match_id", "delta_xT"], ascending=[True, False])
    )

    # team ranking
    team_ranking = (
        df.groupby("team")["delta_xT"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"delta_xT": "total_xT"})
    )

    team_ranking_per_game = (
    df.groupby(["match_id", "team"])["delta_xT"]
    .sum()
    .reset_index()
    .sort_values(["match_id", "delta_xT"], ascending=[True, False])
    )

    return df, player_ranking, player_ranking_per_game, team_ranking, team_ranking_per_game

# main execution function
if __name__ == "__main__":
    df_xt, player_ranking, player_ranking_per_game, team_ranking, team_ranking_per_game = evaluate_xt()

    print("\n===== Top 10 Players by xT contribution =====")
    print(player_ranking.head(10))

    print("\n===== Players by xT contribution per game =====")
    print(player_ranking_per_game.head(10))

    print("\n===== Teams by xT contribution =====")
    print(team_ranking)

    print("\n===== Teams by xT contribution per game =====")
    print(team_ranking_per_game.head(10))

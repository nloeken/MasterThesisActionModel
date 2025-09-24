import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import EVENTS_DIR, THREE_SIXTY_DIR 

# load merge, sort and filter (including only events with freeze_frame) data for specific match
match_id = 3788741 # adjust match_id here
match_events_df = pd.read_json(f'{EVENTS_DIR}/{match_id}.json')
match_360_df = pd.read_json(f'{THREE_SIXTY_DIR}/{match_id}.json')
df = pd.merge(left=match_events_df, right=match_360_df, left_on='id', right_on='event_uuid', how='inner')
df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

# filter: only main events (actions by ball-carrier)
main_event_types = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot']
df = df[df['type'].apply(lambda x: x['name'] in main_event_types)].reset_index(drop=True)

# optional filter: search for specific event (based on 'id')
specific_event_id = "0"  # adjust specific_event_id here
if specific_event_id in df['id'].values:
    initial_idx = df.index[df['id'] == specific_event_id][0]
else:
    print(f"Event '{specific_event_id}' not found. Showing the first event instead.")
    initial_idx = 0

# extract team names and ids 
starting_xi = match_events_df[match_events_df['type'].apply(lambda x: x['name']) == 'Starting XI'].reset_index(drop=True)
team_1_id = starting_xi.loc[0, 'team']['id']
team_1_name = starting_xi.loc[0, 'team']['name']
team_2_id = starting_xi.loc[1, 'team']['id']
team_2_name = starting_xi.loc[1, 'team']['name']

# select reference team (team starting to play on left side of the pitch)
ref_choice = input(f"Which team starts playing from left to right? (Press 1 for {team_1_name} or 2 for {team_2_name}): ")

if ref_choice == '1':
    reference_team_id, reference_team_name = team_1_id, team_1_name
    opponent_team_id, opponent_team_name = team_2_id, team_2_name
elif ref_choice == '2':
    reference_team_id, reference_team_name = team_2_id, team_2_name
    opponent_team_id, opponent_team_name = team_1_id, team_1_name
else:
    raise ValueError("Invalid entry. Only 1 or 2 are allowed.")

# assign zone based on x,y coordinates
def assign_zone(x, y):
    if pd.isna(x) or pd.isna(y):
        return np.nan
    col = int(min(x // (120/5), 4))
    row = int(min(y // (80/4), 3))
    return row * 5 + col

# draw zones on pitch
def draw_zones(ax, grid_x=5, grid_y=4, alpha=0.1, color='grey'):
    zone_width = 120 / grid_x
    zone_height = 80 / grid_y
    for i in range(grid_x):
        for j in range(grid_y):
            rect = mpatches.Rectangle(
                (i*zone_width, j*zone_height),
                zone_width, zone_height,
                linewidth=1, edgecolor='grey', facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)

# draw zone labels
def draw_zone_labels(ax, grid_x=5, grid_y=4):
    zone_width = 120 / grid_x
    zone_height = 80 / grid_y
    for i in range(grid_x):
        for j in range(grid_y):
            zone_id = j * grid_x + i
            ax.text(
                i*zone_width + zone_width/2,
                j*zone_height + zone_height/2,
                str(zone_id),
                ha='center', va='center',
                fontsize=10, color='red', alpha=0.5
            )

# flip coordinates for broadcasting-perspective 
def flip_xy(loc, event_team_id, period):
    if loc is None:
        return None
    x, y = loc
    # if reference_team starts playing on left side: no flipping
    # at halftime: change of sides - flipping
    if (period == 1 and event_team_id != reference_team_id) or (period == 2 and event_team_id == reference_team_id):
        return [120 - x, 80 - y]
    return [x, y]

# plot setup
p = Pitch(pitch_type='statsbomb')
fig, ax = p.draw(figsize=(12, 8))
event_idx = [initial_idx]

def plot_event(idx):
    ax.clear()
    p.draw(ax=ax)

    # draw zones
    draw_zones(ax)
    draw_zone_labels(ax)

    row = df.iloc[idx]
    team_id = row['team']['id']
    event_type = row['type']['name']
    period = row['period']

    # player positions from freeze_frame
    if row.get('freeze_frame') is not None:
        for player in row['freeze_frame']:
            player_loc = flip_xy(player['location'], team_id, period)
            player_team_id = team_id if player['teammate'] else (opponent_team_id if team_id == reference_team_id else reference_team_id)
            color = 'white' if player_team_id == reference_team_id else 'black'
            p.scatter(x=player_loc[0], y=player_loc[1], ax=ax, c=color, s=100, edgecolors='black')

    # ball-carrier position
    if isinstance(row['location'], list):
        loc = flip_xy(row['location'], team_id, period)
        zone_id = assign_zone(loc[0], loc[1])
        p.scatter(x=loc[0], y=loc[1], ax=ax, color='blue', alpha=0.7, s=300, edgecolors='black')
        #ax.text(loc[0]+2, loc[1]+2, f"Zone {zone_id}", color='blue', fontsize=12, fontweight='bold')

    # legend setup
    if period == 1:
        left_team_name, right_team_name = reference_team_name, opponent_team_name
    else:
        left_team_name, right_team_name = opponent_team_name, reference_team_name

    patch_left = mpatches.Patch(color='black' if left_team_name == opponent_team_name else 'white',
                                label=f"{left_team_name}", edgecolor='black')
    patch_right = mpatches.Patch(color='black' if right_team_name == opponent_team_name else 'white',
                                 label=f"{right_team_name}", edgecolor='black')
    ax.legend(handles=[patch_left, patch_right], loc='upper right', fontsize=12, facecolor='lightgrey', edgecolor='black')

    # title setup
    ax.set_title(
        f"Event {idx+1}/{len(df)} | Index: {row['index']} | "
        f"Timestamp: {row['minute']}:{str(row['second']).zfill(2)} | Type: {event_type}",
        fontsize=14
    )

    fig.canvas.draw()

# key event handler
def on_key(event):
    if event.key == 'right':
        event_idx[0] = min(event_idx[0]+1, len(df)-1)
        plot_event(event_idx[0])
    elif event.key == 'left':
        event_idx[0] = max(event_idx[0]-1, 0)
        plot_event(event_idx[0])

fig.canvas.mpl_connect('key_press_event', on_key)

# plot event
plot_event(event_idx[0])
plt.show()

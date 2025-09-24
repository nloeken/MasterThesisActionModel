import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # mpatches is the alias for patches
from scipy.spatial import Voronoi
from config import EVENTS_DIR, THREE_SIXTY_DIR

# === Definition der Zonen-Zentren ===
zone_width = 120 / 5
zone_height = 80 / 4
x_centers = np.linspace(zone_width / 2, 120 - zone_width / 2, 5)
y_centers = np.linspace(zone_height / 2, 80 - zone_height / 2, 4)
xx, yy = np.meshgrid(x_centers, y_centers)
centroids = np.vstack([xx.ravel(), yy.ravel()]).T

# === Funktion zur Zuweisung der Fuzzy-Zonen ===
def calculate_fuzzy_zones(df, x_col='x', y_col='y'):
    valid_loc_df = df.dropna(subset=[x_col, y_col]).copy()
    dist_df = pd.DataFrame(index=valid_loc_df.index)
    degree_df = pd.DataFrame(index=valid_loc_df.index)
    for i, (cx, cy) in enumerate(centroids):
        dist_df[f'zone_dist_{i}'] = np.sqrt((valid_loc_df[x_col] - cx)**2 + (valid_loc_df[y_col] - cy)**2)
    dist_df.replace(0, 1e-9, inplace=True)
    for i in range(len(centroids)):
        sum_of_ratios = sum((dist_df[f'zone_dist_{i}'] / dist_df[f'zone_dist_{j}'])**2 for j in range(len(centroids)))
        degree_df[f'zone_degree_{i}'] = 1 / sum_of_ratios
    zone_series = degree_df.idxmax(axis=1).str.replace('zone_degree_', '').astype(int)
    df['zone'] = zone_series
    return df

# === FINALE, KORRIGIERTE Funktion zum Zeichnen der Zonen ===
def draw_fuzzy_zones(ax, pitch):
    """
    Diese Version umgeht den mplsoccer-Bug, indem sie Matplotlib direkt aufruft.
    """
    points = np.append(centroids, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    voronoi = Voronoi(points)

    for i, region_index in enumerate(voronoi.point_region[:len(centroids)]):
        vertices_indices = voronoi.regions[region_index]
        vertex_list = [voronoi.vertices[v] for v in vertices_indices if v != -1]
        
        if len(vertex_list) >= 3:
            # === HIER IST DIE ÄNDERUNG: Wir umgehen pitch.polygon() ===
            # 1. Erstelle das Polygon-Objekt direkt mit Matplotlib
            polygon = mpatches.Polygon(vertex_list, closed=True, facecolor='grey', edgecolor='black', alpha=0.1, lw=1.5)
            # 2. Füge es direkt zur Achse (ax) hinzu
            ax.add_patch(polygon)
            
    # Zeichne die Zentroiden (Zonenmittelpunkte) und ihre Nummern
    pitch.scatter(centroids[:, 0], centroids[:, 1], ax=ax, c='red', s=50, alpha=0.7)
    for i, (cx, cy) in enumerate(centroids):
        pitch.text(cx, cy, str(i), ax=ax, fontsize=10, color='red',
                   ha='center', va='center', fontweight='bold')


# --- Hauptskript ---

# Daten laden
match_id = 3788741
try:
    match_events_df = pd.read_json(f'{EVENTS_DIR}/{match_id}.json')
    match_360_df = pd.read_json(f'{THREE_SIXTY_DIR}/{match_id}.json')
except FileNotFoundError:
    print(f"Fehler: Daten für match_id {match_id} nicht gefunden.")
    exit()

df = pd.merge(left=match_events_df, right=match_360_df, left_on='id', right_on='event_uuid', how='inner')
df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

# Filtern
main_event_types = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot']
df = df[df['type'].apply(lambda x: x['name'] in main_event_types)].reset_index(drop=True)

# Zonenberechnung
df[['x', 'y']] = pd.DataFrame(df['location'].tolist(), index=df.index)
df = calculate_fuzzy_zones(df, x_col='x', y_col='y')

# Event-Auswahl
specific_event_id = "0"
if specific_event_id in df['id'].values:
    initial_idx = df.index[df['id'] == specific_event_id][0]
else:
    print(f"Event '{specific_event_id}' nicht gefunden. Zeige das erste Event.")
    initial_idx = 0

# Teams
starting_xi = match_events_df[match_events_df['type'].apply(lambda x: x['name'] == 'Starting XI')].reset_index(drop=True)
team_1_id = starting_xi.loc[0, 'team']['id']
team_1_name = starting_xi.loc[0, 'team']['name']
team_2_id = starting_xi.loc[1, 'team']['id']
team_2_name = starting_xi.loc[1, 'team']['name']

# Referenzteam
ref_choice = input(f"Welches Team beginnt von links nach rechts? (1 für {team_1_name} oder 2 für {team_2_name}): ")
if ref_choice == '1':
    reference_team_id, reference_team_name = team_1_id, team_1_name
    opponent_team_id, opponent_team_name = team_2_id, team_2_name
elif ref_choice == '2':
    reference_team_id, reference_team_name = team_2_id, team_2_name
    opponent_team_id, opponent_team_name = team_1_id, team_1_name
else:
    raise ValueError("Ungültige Eingabe.")

# Koordinaten spiegeln
def flip_xy(loc, event_team_id, period):
    if loc is None or not isinstance(loc, list) or len(loc) != 2:
        return None
    x, y = loc
    if (period == 1 and event_team_id != reference_team_id) or (period == 2 and event_team_id == reference_team_id):
        return [120 - x, 80 - y]
    return [x, y]

# Plot-Setup
p = Pitch(pitch_type='statsbomb')
fig, ax = p.draw(figsize=(12, 8))
event_idx = [initial_idx]

def plot_event(idx):
    ax.clear()
    p.draw(ax=ax)
    draw_fuzzy_zones(ax, p)

    row = df.iloc[idx]
    team_id = row['team']['id']
    event_type = row['type']['name']
    period = row['period']
    zone_id = row['zone']

    if row.get('freeze_frame') is not None:
        for player in row['freeze_frame']:
            player_loc = flip_xy(player['location'], team_id, period)
            if player_loc:
                player_team_id = team_id if player['teammate'] else opponent_team_id
                color = 'white' if player_team_id == reference_team_id else 'black'
                p.scatter(x=player_loc[0], y=player_loc[1], ax=ax, c=color, s=100, edgecolors='black')

    loc = flip_xy(row['location'], team_id, period)
    if loc:
        p.scatter(x=loc[0], y=loc[1], ax=ax, color='blue', alpha=0.7, s=300, edgecolors='black', zorder=5)

    if period == 1: left_team_name, right_team_name = reference_team_name, opponent_team_name
    else: left_team_name, right_team_name = opponent_team_name, reference_team_name
    
    patch_left = mpatches.Patch(color='white' if left_team_name == reference_team_name else 'black', label=f"{left_team_name}", edgecolor='black')
    patch_right = mpatches.Patch(color='black' if right_team_name == opponent_team_name else 'white', label=f"{right_team_name}", edgecolor='black')
    ax.legend(handles=[patch_left, patch_right], loc='upper right', fontsize=12, facecolor='lightgrey', edgecolor='black')

    ax.set_title(
        f"Event {idx+1}/{len(df)} | Zone: {int(zone_id) if not pd.isna(zone_id) else 'N/A'} | "
        f"Timestamp: {row['minute']}:{str(row['second']).zfill(2)} | Type: {event_type}",
        fontsize=14
    )
    fig.canvas.draw()

# Steuerung
def on_key(event):
    if event.key == 'right':
        event_idx[0] = min(event_idx[0] + 1, len(df) - 1)
        plot_event(event_idx[0])
    elif event.key == 'left':
        event_idx[0] = max(event_idx[0] - 1, 0)
        plot_event(event_idx[0])

fig.canvas.mpl_connect('key_press_event', on_key)

plot_event(event_idx[0])
plt.show()
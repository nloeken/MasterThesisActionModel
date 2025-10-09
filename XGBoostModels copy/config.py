import os

# paths
# adjust BASE_DIR to local directory, where Statsbomb data is stored
BASE_DIR = '/Users/nloeken/Downloads/open-data'
EVENTS_DIR = os.path.join(BASE_DIR, 'data/events/')
THREE_SIXTY_DIR = os.path.join(BASE_DIR, 'data/three-sixty/')
MERGED_DIR = os.path.join(BASE_DIR, 'merged/')  
META_DIR = os.path.join(BASE_DIR, 'meta/')
COMBINED_FILE = os.path.join(BASE_DIR, 'combined/combined_all.csv')
SAMPLE_FILE = os.path.join(BASE_DIR, 'combined/combined_sample.csv')
PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/xgboost_predictions.csv')
SAMPLE_PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/xgboost_sample_predictions.csv')

# constants
XT_MATRIX = 'https://karun.in/blog/data/open_xt_12x8_v1.json'
MAX_EVENT_GAP = 10  # in seconds
MAIN_EVENT_TYPES = [
    'Half Start', 'Pass', 'Clearance', 'Carry',
    'Duel', 'Dribble', '50/50', 'Shot'
]
centroid_x_100 = [8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 33.5, 66.5, 33.5, 66.5, 33.5, 66.5, 8.5, 91.5]
centroid_y_100 = [89.45, 89.45, 89.45, 89.45, 89.45, 89.45, 10.55, 10.55, 10.55, 10.55, 10.55, 10.55, 71.05, 71.05, 50., 50., 28.95, 28.95, 50., 50.]

# create directories
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)

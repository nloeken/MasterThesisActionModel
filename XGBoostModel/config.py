import os

# Paths
BASE_DIR = '/Users/nloeken/Desktop//open-data'
EVENTS_DIR = os.path.join(BASE_DIR, 'data/events/')
THREE_SIXTY_DIR = os.path.join(BASE_DIR, 'data/three-sixty/')
MERGED_DIR = os.path.join(BASE_DIR, 'merged/')
COMBINED_FILE = os.path.join(BASE_DIR, 'combined/combined_all.csv')
SAMPLE_FILE = os.path.join(BASE_DIR, 'combined/combined_sample.csv')

# Constants
MAX_EVENT_GAP = 10  # in seconds, adjust threshold here
MAIN_EVENT_TYPES = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot'] # adjust included event types here

# Create directories
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)



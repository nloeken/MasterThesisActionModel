import os
# noch anzupassen, um hardcoded Pfade zu vermeiden
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVENTS_DIR = os.path.join(BASE_DIR, 'open-data', 'data', 'events')
THREE_SIXTY_DIR = os.path.join(BASE_DIR, 'open-data', 'data', 'three-sixty')
MERGED_DIR = os.path.join(BASE_DIR, 'open-data', 'merged')
COMBINED_FILE = os.path.join(BASE_DIR, 'open-data', 'combined', 'contextualevents_all.csv')
SAMPLE_FILE = os.path.join(BASE_DIR, 'open-data', 'combined', 'contextualevents_sample.csv')

# Create directories
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)

# Define main event types
MAIN_EVENT_TYPES = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot']

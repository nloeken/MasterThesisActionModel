import ast
import numpy as np

def extract_name(field):
    if isinstance(field, dict):
        return field.get("name")
    return field

# helper to get JSON-like strings from CSV
def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x

# helper to safe location data
def safe_location(loc):
    if isinstance(loc, list) and len(loc) == 2:
        return loc
    return [None, None]

# helper to estimate game phase
def get_match_phase(minute):
    if minute < 15: return "early_first"
    elif minute < 30: return "mid_first"
    elif minute < 45: return "late_first"
    elif minute < 60: return "early_second"
    elif minute < 75: return "mid_second"
    else: return "late_second"

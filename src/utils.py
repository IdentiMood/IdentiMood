import json

KEY_ESC = 27
KEY_ENTER = 32
KEY_SPACE = 13

OPERATION_VERIFY_IDENTITY = 0
OPERATION_VERIFY_MOOD = 1

OPERATION_ENROLLMENT_IDENTITY = 2
OPERATION_ENROLLMENT_MOOD = 3

OPERATIONS_WINDOW_LABELS = [
    "Provide a neutral face",
    "Now express your chosen mood",
    "Provide a neutral face",
    "Select your favorite mood",
]
OPERATIONS_WINDOW_TITLES = [
    "[IdentiMood] Identity verification",
    "[IdentiMood] Mood verification",
    "[IdentiMood] Enrollment - Identity",
    "[IdentiMood] Enrollment - Mood",
]


def load_config():
    with open("./config.json", "r") as f:
        config = json.load(f)
    return config

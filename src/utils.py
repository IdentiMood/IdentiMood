import json
import os
import pwd

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
    """
    Loads the configuration from the ./config.json file
    """
    with open("./config.json", "r", encoding="utf8") as f:
        config = json.load(f)
    return config


def get_identity(argv: list) -> str:
    """
    Gets the identity to user for the recognition operations.
    If a string has been supplied (eg. `python3 main.py john`), then it will be used.
    Else, the current logged-in user will be used.

    This function is supposed to be called by passing to it the argument vector (sys.argv).
    """
    if len(argv) == 1:
        # return the currently logged-in user
        return pwd.getpwuid(os.getuid()).pw_name
    return argv[1]

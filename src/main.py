import sys
import tempfile
import json
from utils import *
from operations import Operations
from window import Window
import cv2


class App:
    def __init__(self, config, claimed_identity):
        self.config = config
        self.claimed_identity = claimed_identity
        self.operations = Operations(config)

    def authenticate(self) -> bool:
        identity_verified, aborted = self.show_photo_window(OPERATION_VERIFY_IDENTITY)
        if aborted:
            return False

        mood_verified, _ = self.show_photo_window(OPERATION_VERIFY_MOOD)
        return identity_verified and mood_verified

    def show_photo_window(self, operation: int) -> (bool, bool):
        window = Window(operation)
        if window.shot_button_pressed:
            return self.handle_probe(operation, window.frame), False
        return False, True

    def handle_probe(self, operation: int, frame) -> (bool, bool):
        verified = False
        try:
            self.operations.detect_face(frame)
        except ValueError as error:
            print(
                "Error while handling the probe.",
                error,
                file=sys.stderr,
            )
            return False

        if operation == OPERATION_VERIFY_IDENTITY:
            verified = self.operations.verify_identity(frame, self.claimed_identity)
        elif operation == OPERATION_VERIFY_MOOD:
            verified = self.operations.verify_mood(frame, self.claimed_identity)

        return verified


if __name__ == "__main__":
    config = load_config()

    if len(sys.argv) == 1:
        claimed_identity = "me"
    else:
        claimed_identity = sys.argv[1]

    app = App(load_config(), claimed_identity)

    print("Is it really you? Please confirm your identity.")
    opt = input("[y]/n: ")

    if opt == "" or opt == "y":
        authenticated = app.authenticate()
        print("Authenticated:", authenticated)
    else:
        exit(1)

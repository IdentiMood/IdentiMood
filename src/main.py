import sys
import tempfile
import json
from utils import *
from operations import Operations
from window import Window
import cv2


class App:
    """
    App is the main coordinator for the IdentiMood application.
    """

    def __init__(self, config, claimed_identity):
        self.config = config
        self.claimed_identity = claimed_identity
        self.operations = Operations(config)

        if not self.operations.is_user_enrolled(claimed_identity):
            print(f"User {claimed_identity} is not enrolled.", file=sys.stderr)
            sys.exit(1)

    def authenticate(self) -> bool:
        """
        Starts the authentication procedure, by verifying identity and mood in two separate steps.
        """
        identity_verified, aborted = self.show_photo_window(OPERATION_VERIFY_IDENTITY)
        if aborted:
            return False

        mood_verified, _ = self.show_photo_window(OPERATION_VERIFY_MOOD)
        return identity_verified and mood_verified

    def show_photo_window(self, operation: int) -> (bool, bool):
        """
        Shows a Window to shot the picture.
        Returns a tuple (identity_verified, operation_has_been_aborted).
        """
        window = Window(operation)
        if window.shot_button_pressed:
            return self.handle_probe(operation, window.frame), False
        return False, True

    def handle_probe(self, operation: int, frame) -> (bool, bool):
        """
        Calls the given recognition operation on the given frame.
        Returns True if the result of the operation was successful.
        """
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
    claimed_identity = get_identity(sys.argv)
    app = App(load_config(), claimed_identity)

    print("Is it really you? Please confirm your identity.")
    opt = input("[y]/n: ")

    if opt == "" or opt == "y":
        authenticated = app.authenticate()
        print("Authenticated:", authenticated)
    else:
        exit(1)

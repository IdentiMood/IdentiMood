import sys
import tempfile
import json
from utils import *
from operations import verify_identity, verify_mood
from window import Window
import cv2


class App:
    def __init__(self, config, claimed_identity):
        self.config = config
        self.claimed_identity = claimed_identity

    def authenticate(self) -> bool:
        identity_verified, aborted = self.show_photo_window(OPERATION_VERIFY_IDENTITY)
        if aborted:
            return False

        mood_verified, _ = self.show_photo_window(OPERATION_VERIFY_MOOD)
        return identity_verified and mood_verified

    def show_photo_window(self, operation: int) -> (bool, bool):
        window = Window(operation)
        if window.shot_button_pressed:
            return self.save_and_verify(operation, window.frame), False
        return False, True

    def save_and_verify(self, operation: int, frame) -> (bool, bool):
        verified = False
        tmp = tempfile.NamedTemporaryFile(prefix="identimood", suffix=".jpg")
        try:
            cv2.imwrite(tmp.name, frame)
            if operation == OPERATION_VERIFY_IDENTITY:
                verified = verify_identity(
                    tmp.name, self.claimed_identity, self.config["verify"]
                )
            elif operation == OPERATION_VERIFY_MOOD:
                verified = verify_mood(
                    tmp.name, self.claimed_identity, self.config["mood"]
                )
        except ValueError as e:
            verified = False
            print("Error while handling the probe.", e)
        finally:
            tmp.close()

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

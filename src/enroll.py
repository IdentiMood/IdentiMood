import sys
import cv2
from utils import OPERATION_ENROLLMENT_MOOD, OPERATION_ENROLLMENT_IDENTITY, load_config
from operations import Operations
from window import Window
from main import App


class Enroller:
    def __init__(self):
        self.operations = Operations(config)

        if identity in self.operations.get_enrolled_identities():
            authenticated = self._ask_authentication()
            if not authenticated:
                print("Authentication failed", file=sys.stderr)
                exit(1)
            print("Authenticated")

        frame_identity, aborted = self.show_window(OPERATION_ENROLLMENT_IDENTITY)
        if aborted or frame_identity is None:
            print("Authentication failed", file=sys.stderr)
            exit(1)

        frame_mood, aborted = self.show_window(OPERATION_ENROLLMENT_MOOD)
        if aborted or frame_mood is None:
            print("Authentication failed", file=sys.stderr)
            exit(1)

        mood = self.extract_mood(frame_mood)
        if mood is None:
            print("Authentication failed", file=sys.stderr)
            exit(1)

        self.operations.save_template(frame_identity, identity, preprocess=True)
        self.operations.save_mood(identity, mood)

    def show_window(self, operation: int):
        window = Window(operation)
        if window.shot_button_pressed:
            return window.frame, False
        return None, True

    def extract_mood(self, frame) -> str:
        mood = None
        try:
            mood = self.operations.get_mood(frame)
        except ValueError as e:
            print("Error while handling the probe.", e, file=sys.stderr)
            mood = None

        return mood

    def _ask_authentication(self):
        print("User already exists. Do you want to add another template?")
        opt = input("[y]/n: ")

        if opt == "" or opt == "y":
            app = App(config, identity)
            return app.authenticate()
        return False


if __name__ == "__main__":
    config = load_config()

    if len(sys.argv) == 1:
        identity = "me"
    else:
        identity = sys.argv[1]

    enroller = Enroller()

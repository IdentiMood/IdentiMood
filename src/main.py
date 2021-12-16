import sys
import tempfile
import json
from utils import *
from operations import verify_identity, verify_mood
import cv2


class App:
    config = dict()
    claimed_identity = ""

    def __init__(self, config, claimed_identity):
        self.config = config
        self.claimed_identity = claimed_identity

    def show_photo_window(self, operation: int) -> (bool, bool):
        aborted = False
        verified = False

        video = cv2.VideoCapture(0)
        while True:
            _, frame = video.read()
            cv2.imshow("Show your face", frame)

            key = cv2.waitKey(1)
            if key == KEY_ESC:
                aborted = True
                break
            if key in (KEY_ENTER, KEY_SPACE):
                tmp = tempfile.NamedTemporaryFile(prefix="identimood", suffix=".jpg")
                try:
                    cv2.imwrite(tmp.name, frame)
                    if operation == OPERATION_VERIFY_IDENTITY:
                        verified = verify_identity(
                            tmp.name, claimed_identity, self.config["verify"]
                        )
                    elif operation == OPERATION_VERIFY_MOOD:
                        verified = verify_mood(
                            tmp.name, claimed_identity, self.config["mood"]
                        )
                    break
                except ValueError as e:
                    verified = False
                    print("Error while handling the probe.", e)
                finally:
                    tmp.close()

        video.release()
        cv2.destroyAllWindows()
        return verified, aborted


def load_config():
    with open("./config.json", "r") as f:
        config = json.load(f)
    return config


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
        identity_verified, aborted = app.show_photo_window(OPERATION_VERIFY_IDENTITY)
        if aborted:
            exit(1)

        mood_verified, _ = app.show_photo_window(OPERATION_VERIFY_MOOD)
        print(f"Verified: {identity_verified and mood_verified}")
    else:
        exit(1)

import os
import sys
import json
import uuid
import cv2
from deepface import DeepFace


class Operations:
    def __init__(self, config):
        self.gallery_path = config["gallery_path"]
        self.config = config

    def get_enrolled_identities(self) -> list:
        return os.listdir(self.gallery_path)

    def get_gallery_templates(self, identity_claim: str) -> list:
        base_path = os.path.join(self.gallery_path, identity_claim)
        files = []
        for f in os.listdir(base_path):
            if f.endswith(".png") or f.endswith(".jpg"):
                files.append(os.path.join(base_path, f))
        return files

    def load_meta(self, identity_claim: str) -> dict:
        meta_path = os.path.join(self.gallery_path, identity_claim, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta

    def save_template(self, frame, identity: str, preprocess=True):
        if preprocess:
            try:
                frame = DeepFace.detectFace(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.convertScaleAbs(frame, alpha=(255.0))
            except ValueError as error:
                print("Error while preprocessing the frame.", error, file=sys.stderr)

        filename = str(uuid.uuid4()) + ".jpg"
        basepath = os.path.join(self.gallery_path, identity)
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        path = os.path.join(basepath, filename)
        cv2.imwrite(path, frame)

    def save_mood(self, identity: str, mood: str):
        basepath = os.path.join(self.gallery_path, identity)
        meta_path = os.path.join(self.gallery_path, identity, "meta.json")
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        if not os.path.exists(meta_path):
            meta = self._make_empty_meta()
        else:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        meta["favorite_mood"] = mood
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta))

    def verify_identity(self, probe, identity_claim: str) -> bool:
        files = self.get_gallery_templates(identity_claim)
        results = []
        for template in files:
            print(f"Verifying probe against template {template}...")
            try:
                result = DeepFace.verify(
                    img1_path=probe,
                    img2_path=template,
                    model_name=self.config["verify"]["model_name"],
                    detector_backend=self.config["verify"]["detector_backend"],
                    distance_metric=self.config["verify"]["distance_metric"],
                    normalization=self.config["verify"]["normalization"],
                )
            except ValueError as error:
                print(
                    "Error while handling the probe or gallery template.",
                    error,
                    file=sys.stderr,
                )
            except AttributeError as error:
                # happens when the loaded img is a "NoneType" and NumPy operations fail
                print(
                    "Error while handling the gallery template.", error, file=sys.stderr
                )
            else:
                results.append(result)

        results.sort(key=lambda r: r["distance"])
        return results[0]["distance"] < self.config["verify"]["threshold"]

    def detect_face(self, probe):
        return DeepFace.detectFace(
            probe, detector_backend=self.config["verify"]["detector_backend"]
        )

    def verify_mood(self, probe, identity_claim: str) -> bool:
        favorite_mood = self.load_meta(identity_claim)["favorite_mood"]

        print("Finding the probe's mood")
        result = DeepFace.analyze(
            probe,
            actions=["emotion"],
            detector_backend=self.config["mood"]["detector_backend"],
        )

        return (
            result["emotion"][favorite_mood] >= self.config["mood"]["threshold_percent"]
        )

    def get_mood(self, probe) -> str:
        result = DeepFace.analyze(
            probe,
            actions=["emotion"],
            detector_backend=self.config["mood"]["detector_backend"],
        )
        return result["dominant_emotion"]

    def _make_empty_meta(self) -> object:
        return {"name": "", "favorite_mood": ""}

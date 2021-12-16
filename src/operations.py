import os
import json
from deepface import DeepFace

GALLERY_PATH = "./gallery"


def get_gallery_templates(identity_claim: str) -> list:
    base_path = os.path.join(GALLERY_PATH, identity_claim)
    files = []
    for f in os.listdir(base_path):
        if f.endswith(".png") or f.endswith(".jpg"):
            files.append(os.path.join(base_path, f))
    return files


def load_meta(identity_claim: str) -> dict:
    meta_path = os.path.join(GALLERY_PATH, identity_claim, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta


def verify_identity(probe_path: str, identity_claim: str, config: dict) -> bool:
    files = get_gallery_templates(identity_claim)
    results = []
    for template in files:
        print(f"Verifying {probe_path} against template {template}...")
        result = DeepFace.verify(
            img1_path=probe_path,
            img2_path=template,
            model_name=config["model_name"],
            detector_backend=config["detector_backend"],
            distance_metric=config["distance_metric"],
            normalization=config["normalization"],
        )
        results.append(result)

    results.sort(key=lambda r: r["distance"])
    return results[0]["distance"] < config["threshold"]


def verify_mood(probe_path: str, identity_claim: str, config: dict) -> bool:
    favorite_mood = load_meta(identity_claim)["favorite_mood"]

    print(f"Finding {probe_path}'s mood")
    result = DeepFace.analyze(
        probe_path, actions=["emotion"], detector_backend=config["detector_backend"]
    )

    return result["emotion"][favorite_mood] >= config["threshold_percent"]

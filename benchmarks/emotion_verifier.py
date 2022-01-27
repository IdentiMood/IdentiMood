# from deepface import DeepFace
import argparse
from deepface import DeepFace
import os
import numpy as np
import time
from datetime import datetime
import json
from tqdm import tqdm

errors = []

DEFAULT_MODELS_MASK = "10000000"

models_list = [
    "VGG-Face",
    "OpenFace",
    "Facenet",
    "Facenet512",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="File containing the dataset's file paths")
parser.add_argument(
    "-e",
    "--evaluation-method",
    help=" \
        0 --> actual emotion should be the same of DeepFace predominant \
        emotion. \
        1 --> 0 + relative distance from prevalent and second prevalent \
        emotion must be greater than a threshold",
    default=0,
    type=int,
)
parser.add_argument(
    "-l",
    "--limit",
    help="Whether to limit the (eventually shuffled) paths list to a certain\
        length",
    type=int,
)
parser.add_argument(
    "-tr",
    "--threshold-range",
    nargs="+",
    action="extend",
    help="creates an evenly spaced range over a specified interval. \
        Args: start, stop, number of values",
    type=float,
)
parser.add_argument(
    "-m",
    "--models-mask",
    help="Binary mask to decide what Deep Learning models to use for face\
    verification. Mask index meaning: 0 --> VGG, 1 --> OpenFace, 2--> Facenet,\
    3 --> Facenet512, 4 --> Facebook DeepFace, 5 --> DeepID, 6 --> Dlib, 7 -->\
    ArcFace",
    type=str,
)
parser.add_argument(
    "-min",
    "--minimal-output",
    help="Whether to have a small minimal output (progress bar only)",
    action="store_true",
)

args = parser.parse_args()

if args.threshold_range:
    args.thresholds = np.linspace(
        args.threshold_range[0], args.threshold_range[1], int(args.threshold_range[2])
    )
else:
    args.thresholds = [0]

if not (set(args.models_mask).issubset({"0", "1"}) and bool(args.models_mask)):
    args.models_mask = DEFAULT_MODELS_MASK
    print("Provided models mask not binary, defaulting to", DEFAULT_MODELS_MASK)

if len(args.models_mask) != len(models_list):
    args.models_mask = DEFAULT_MODELS_MASK
    print("Provided models mask not complete, defaulting to", DEFAULT_MODELS_MASK)

models_dict = {}

for (i, j) in zip(list(range(0, len(args.models_mask))), models_list):
    if args.models_mask[i] == "1":
        models_dict[j] = DeepFace.build_model(j)


analyze_output_hardcoded = {
    "dominant_emotion": "neutral",
    "emotion": {
        "sad": 37.65260875225067,
        "angry": 0.15512987738475204,
        "surprise": 0.0022171278033056296,
        "fear": 1.2489334680140018,
        "happy": 4.609785228967667,
        "disgust": 9.698561953541684e-07,
        "neutral": 56.33133053779602,
    },
}

TUTFS_emotion_codes = ["neutral", "neutral", "happy", "neutral", "surprise", "neutral"]

KDEF_emotion_codes = {
    "AF": "fear",
    "AN": "angry",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise",
}

yalefaces_neutral_cases = [
    "glasses",
    "leftlight",
    "centerlight",
    "rightlight",
    "noglasses",
    "normal",
    "sleepy",
    "wink",
    "gif",
]

with open(args.input) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    if args.limit is not None:
        lines = lines[: args.limit]
    total_lines = len(lines) * len(args.thresholds) * len(models_dict.keys())


def yalefaces_actual_emotion(file_name):
    emotion = file_name.split(".")[-3]

    if any(x in emotion for x in yalefaces_neutral_cases):
        return "neutral"
    elif emotion == "surprised":
        return "surprise"
    else:
        return emotion


def is_TUTFS(folder_name):
    return folder_name.isdigit()


def is_yalefaces(folder_name):
    return "subject" in folder_name


def verify_emotion_thresholds(
    analyze_output,
    actual_emotion,
    threshold,
    first_emotion_score=None,
    second_emotion_score=None,
):
    bool_emotion_match = analyze_output["dominant_emotion"] == actual_emotion

    if args.evaluation_method == 0:
        return bool_emotion_match
    if args.evaluation_method == 1:
        return (
            bool_emotion_match
            and abs(first_emotion_score - second_emotion_score) >= threshold
        )


def compute_emotions_against_thresholds(thresholds=[0]):

    errors_count = dict()

    for model in models_dict.keys():
        errors_count[model] = dict()

        for threshold in thresholds:
            errors_count[model][threshold] = {"correct": 0, "wrong": 0}

    current_combination = 1

    for line in tqdm(lines):
        for model in models_dict.keys():

            try:
                analyze_output = DeepFace.analyze(line, ["emotion"])
            except ValueError as e:
                print(e)

                errors.append({"error": str(e), "img_path": line})

                continue

            dominant_emotion = analyze_output["dominant_emotion"]
            first_emotion_score = float(analyze_output["emotion"][dominant_emotion])
            sorted_emotion_dict = dict(
                sorted(analyze_output["emotion"].items(), key=lambda item: item[1])
            )
            second_emotion = list(sorted_emotion_dict.keys())[1]
            second_emotion_score = float(sorted_emotion_dict[second_emotion])

            folder_name = os.path.basename(os.path.dirname(line))

            if is_TUTFS(folder_name):
                actual_emotion = TUTFS_emotion_codes[int(os.path.basename(line)[-5])]
            elif is_yalefaces(folder_name):
                actual_emotion = yalefaces_actual_emotion(line)
            else:
                actual_emotion = KDEF_emotion_codes[os.path.basename(line)[-7:-5]]

            for threshold in thresholds:
                verified = verify_emotion_thresholds(
                    analyze_output,
                    actual_emotion,
                    threshold,
                    first_emotion_score,
                    second_emotion_score,
                )

                if verified:
                    errors_count[model][threshold]["correct"] += 1
                else:
                    errors_count[model][threshold]["wrong"] += 1

                if not args.minimal_output:
                    print(f"Analyzing file: {line}")
                    print(f"Model:          {model}")
                    print(f"DeepFace says:  {analyze_output['dominant_emotion']}")
                    print(f"Second emotion: {second_emotion}")
                    print(f"Should be:      {actual_emotion}")
                    print(f"Threshold:      {threshold}")
                    print(
                        f"Progress:             {current_combination}/{total_lines} \
                        [{round(current_combination/total_lines * 100, 2)}%]"
                    )
                    print("----------------------------------------------------------")

                current_combination += 1

    return errors_count


results = compute_emotions_against_thresholds(args.thresholds)

file_name = datetime.fromtimestamp(time.time()).strftime("%y_%m_%d_%H-%M-%S")

with open("../logs/emotion/" + file_name + ".json", "w") as output_log:
    output_log.write(json.dumps(results, indent=4, cls=NpEncoder))
    output_log.close()

with open("../logs/emotion/" + file_name + ".err.json", "w") as error_log:
    error_log.write(json.dumps(errors, indent=4, cls=NpEncoder))
    error_log.close()

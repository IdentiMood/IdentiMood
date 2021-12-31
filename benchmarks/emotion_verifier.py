# from deepface import DeepFace
import argparse
from deepface import DeepFace
import os
import numpy as np
import time
from datetime import datetime
import json

errors = []


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

args = parser.parse_args()

if args.threshold_range:
    args.thresholds = np.linspace(
        args.threshold_range[0], args.threshold_range[1], int(args.threshold_range[2])
    )
else:
    args.thresholds = [0]


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

with open(args.input) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    if args.limit is not None:
        lines = lines[: args.limit]
    total_lines = len(lines) * len(args.thresholds)


def is_TUTFS(folder_name):
    return folder_name.isdigit()


def verify_emotion_thresholds(analyze_output, actual_emotion, threshold):
    dominant_emotion = analyze_output["dominant_emotion"]
    return (analyze_output["emotion"][actual_emotion] / 100) >= threshold


def compute_emotions_against_thresholds(thresholds=[0]):
    errors_count = dict()

    current_combination = 1

    for threshold in thresholds:
        errors_count[threshold] = {"correct": 0, "wrong": 0}

    for line in lines:

        try:
            analyze_output = DeepFace.analyze(line, ["emotion"])
        except ValueError as e:
            print(e)

            errors.append(
                {
                    "error": str(e),
                    "img_path": line,
                }
            )

            continue

        folder_name = os.path.basename(os.path.dirname(line))

        if is_TUTFS(folder_name):
            actual_emotion = TUTFS_emotion_codes[int(os.path.basename(line)[-5])]
        else:
            actual_emotion = KDEF_emotion_codes[os.path.basename(line)[-7:-5]]

        for threshold in thresholds:
            verified = verify_emotion_thresholds(
                analyze_output, actual_emotion, threshold
            )
            if verified:
                errors_count[threshold]["correct"] += 1
            else:
                errors_count[threshold]["wrong"] += 1

            print(f"Analyzing file:  {line}")
            print(f"DeepFace says:   {analyze_output['dominant_emotion']}")
            print(f"Should be:       {actual_emotion}")
            print(f"Threshold:       {threshold}")
            print(
                f"Progress:        {current_combination}/{total_lines} \
                [{round(current_combination/total_lines * 100, 2)}%]"
            )
            print(
                "----------------------------------------------------------------------------------------"
            )

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

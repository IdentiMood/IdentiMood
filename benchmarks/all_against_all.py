import argparse
import random
from deepface import DeepFace
import json
import time
from datetime import datetime
from plotter import *
import numpy as np
import os
from joblib import Parallel, delayed

errors = []

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="File containing the dataset's file paths")
parser.add_argument("-s", "--shuffle", help="Whether to shuffle the paths list", action="store_true")
parser.add_argument("-l", "--limit", help="Whether to limit the (eventually shuffled) paths list to a certain length", type=int)
parser.add_argument("-t", "--thresholds", nargs="+", action="extend", help="A list of thresholds to test", type=float)
parser.add_argument("-tr", "--threshold-range", nargs="+", action="extend", help="creates an evenly spaced range over a specified interval. Args: start, stop, number of values", type=float)
parser.add_argument("-dc", "--cosine", help="Test with the cosine distance metric", action="store_true")
parser.add_argument("-de", "--euclidean", help="Test with the euclidean distance metric", action="store_true")
parser.add_argument("-del2", "--euclidean_l2", help="Test with the euclidean_l2 distance metric", action="store_true")
parser.add_argument("-v", "--verbose", help="Print iteration results", action="store_true")
args = parser.parse_args()

if args.threshold_range:
    args.thresholds = np.linspace(
        args.threshold_range[0], args.threshold_range[1], int(args.threshold_range[2])
    )

distance_metrics = []
if args.cosine:
    distance_metrics.append("cosine")
if args.euclidean:
    distance_metrics.append("euclidean")
if args.euclidean_l2:
    distance_metrics.append("euclidean_l2")

model = DeepFace.build_model('VGG-Face')

def verify(distance_score, threshold):
    return distance_score <= threshold

with open(args.input) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    if args.shuffle:
        random.shuffle(lines)
    if args.limit is not None:
        lines = lines[:args.limit]
    total_combinations = (len(lines)**2 - len(lines)) * len(args.thresholds) * len(distance_metrics)

genuine_acceptances = dict()
genuine_rejections = dict()
false_acceptances = dict()
false_rejections = dict()

genuine_attempts = dict()
impostor_attempts = dict()

genuine_acceptance_rate = dict()
genuine_rejection_rate = dict()
false_acceptance_rate = dict()
false_rejection_rate = dict()
error_rate = dict()

results = dict()

def perform_all_against_all(distance_metrics = [], thresholds = [], verbose = False):

    current_combination = 1

    if (verbose):
        total_combinations_w_threshold = 250000 # number of verifies times number of metrics
        print("Total number of input files", len(lines))
        print("One physical machine will compute", total_combinations_w_threshold, "combinations")
        print("One physical machine will take", total_combinations_w_threshold * 0.6 , "secs")
        print("One physical machine will take", total_combinations_w_threshold * 0.6 / 60, "mins")
        print("One physical machine will take", total_combinations_w_threshold * 0.6 / 3600, "hours")
        print("One physical machine will take", total_combinations_w_threshold * 0.6 / 86400, "days")
        print("----------------------------------------------------------------------------------------")

        total_combinations = 250000 # number of verifies times number of metrics
        print("Total number of input files", len(lines))
        print("One physical machine will compute", total_combinations, "combinations")
        print("One physical machine will take", total_combinations * 0.6 / len(args.thresholds), "secs")
        print("One physical machine will take", total_combinations * 0.6 / 60 / len(args.thresholds), "mins")
        print("One physical machine will take", total_combinations * 0.6 / 3600 / len(args.thresholds), "hours")
        print("One physical machine will take", total_combinations * 0.6 / 86400 / len(args.thresholds), "days")
        print("----------------------------------------------------------------------------------------")

    exit()

    for metric in distance_metrics:
        genuine_acceptances[metric] = dict()
        genuine_rejections[metric] = dict()
        false_acceptances[metric] = dict()
        false_rejections[metric] = dict()

        genuine_attempts[metric] = dict()
        impostor_attempts[metric] = dict()

        genuine_acceptance_rate[metric] = dict()
        genuine_rejection_rate[metric] = dict()
        false_acceptance_rate[metric] = dict()
        false_rejection_rate[metric] = dict()
        error_rate[metric] = dict()

        for threshold in thresholds:
            threshold_str = str(threshold)

            genuine_acceptances[metric][threshold_str] = 0
            genuine_rejections[metric][threshold_str] = 0
            false_acceptances[metric][threshold_str] = 0
            false_rejections[metric][threshold_str] = 0

            genuine_attempts[metric][threshold_str] = 0
            impostor_attempts[metric][threshold_str] = 0

            genuine_acceptance_rate[metric][threshold_str] = 0
            genuine_rejection_rate[metric][threshold_str] = 0
            false_acceptance_rate[metric][threshold_str] = 0
            false_rejection_rate[metric][threshold_str] = 0
            error_rate[metric][threshold_str] = 0

            for first_identity_index in range(0, len(lines)):
                first_identity_name = lines[first_identity_index].split('/')[3]

                for second_identity_index in range(0, len(lines)):

                    start_time = time.time()

                    if (second_identity_index == first_identity_index): continue

                    try:
                        result = DeepFace.verify(
                            img1_path = lines[first_identity_index],
                            img2_path = lines[second_identity_index],
                            model = model
                        )
                    except ValueError as e:
                        # happens when the img cannot be loaded
                        print(e, lines[first_identity_index], lines[second_identity_index])

                        errors.append({
                            "error": e,
                            "img1_path": lines[first_identity_index],
                            "img2_path": lines[second_identity_index],
                        })
                        continue

                    second_identity_name = lines[second_identity_index].split('/')[3]

                    if (first_identity_name == second_identity_name):
                        genuine_attempts[metric][threshold_str] += 1
                    else:
                        impostor_attempts[metric][threshold_str] += 1

                    verified = verify(result['distance'], threshold)

                    genuine_acceptances[metric][threshold_str] += int(verified and (first_identity_name == second_identity_name))

                    genuine_rejections[metric][threshold_str] += int(not verified and not (first_identity_name == second_identity_name))

                    false_acceptances[metric][threshold_str] += int(verified and not (first_identity_name == second_identity_name))

                    false_rejections[metric][threshold_str] += int(not verified and (first_identity_name == second_identity_name))

                    if (verbose):
                        print(f"Matching faces:  {first_identity_index} ({first_identity_name}) VS {second_identity_index} ({second_identity_name})")
                        print(f"DeepFace says:   {verified}")
                        print(f"Should be:       {first_identity_name == second_identity_name}")
                        print(f"Distance:        {result['distance']}")
                        print(f"Threshold:       {threshold}")
                        print(f"Distance metric: {metric}")
                        print(f"Progress:        {current_combination}/{total_combinations} [{round(current_combination/total_combinations*100, 2)}%]")
                        print(f"Execution time:  {time.time() - start_time} seconds")
                        print("----------------------------------------------------------------------------------------")

                        current_combination += 1
            ga = genuine_attempts[metric][threshold_str]

            if (ga != 0):
                genuine_acceptance_rate[metric][threshold_str] = \
                    genuine_acceptances[metric][threshold_str] / ga

                false_rejection_rate[metric][threshold_str] = \
                    false_rejections[metric][threshold_str] / ga

            ia = impostor_attempts[metric][threshold_str]

            if (ia != 0):
                genuine_rejection_rate[metric][threshold_str] = \
                    genuine_rejections[metric][threshold_str] / ia

                false_acceptance_rate[metric][threshold_str] = \
                    false_acceptances[metric][threshold_str] / ia

            # should never be the case, but, one can never know... :(
            if (ga + ia != 0):
                error_rate[metric][threshold_str] = (
                    false_acceptances[metric][threshold_str] +
                    false_rejections[metric][threshold_str]
                ) / (ga + ia)

    return {
        "genuine_acceptances": genuine_acceptances,
        "genuine_rejections": genuine_rejections,
        "false_acceptances": false_acceptances,
        "false_rejections": false_rejections,
        "genuine_attempts": genuine_attempts,
        "impostor_attempts": impostor_attempts,
        "genuine_acceptance_rate": genuine_acceptance_rate,
        "genuine_rejection_rate": genuine_rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "false_rejection_rate": false_rejection_rate,
        "error_rate": error_rate
    }

def print_recognition_metrics():
    for metric in distance_metrics:
        for threshold in args.thresholds:
            threshold_str = str(threshold)

            # print("GA[", threshold_str, "]: ", genuine_acceptances[metric][threshold_str], genuine_attempts[metric][threshold_str])
            # print("GR[", threshold_str, "]: ", genuine_rejections[metric][threshold_str], genuine_attempts[metric][threshold_str])
            # print("FA[", threshold_str, "]: ", false_acceptances[metric][threshold_str], impostor_attempts[metric][threshold_str])
            # print("FR[", threshold_str, "]: ", false_rejections[metric][threshold_str], impostor_attempts[metric][threshold_str])

            print("-----------")

            print(
                "Genuine attemps[", metric, "][", threshold_str, "]: ",
                genuine_attempts[metric][threshold_str]
            )
            print(
                "Impostor attemps[", metric, "][", threshold_str, "]: ",
                impostor_attempts[metric][threshold_str]
            )
            print()
            print(
                "GAR[", metric, "][", threshold_str, "]: ",
                genuine_acceptance_rate[metric][threshold_str]
            )
            print(
                "GRR[", metric, "][", threshold_str, "]: ",
                genuine_rejection_rate[metric][threshold_str]
            )
            print(
                "FAR[", metric, "][", threshold_str, "]: ",
                false_acceptance_rate[metric][threshold_str]
            )
            print(
                "FRR[", metric, "][", threshold_str, "]: ",
                false_rejection_rate[metric][threshold_str]
            )
            print(
                "Error rate[", metric, "][", threshold_str, "]: ", error_rate[metric][threshold_str]
            )
            print()

# TFF: x = thresholds, y = FRR, FAR
def compute_threshold_FAR_FRR_plot(distance_metric_name, show_plot, file_name):

    folder_path = f'''../plots/{distance_metric_name}/{args.threshold_range[0]}_
        {args.threshold_range[1]}_{int(args.threshold_range[2])}'''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plot_name = f"{folder_path}/tff_{file_name}.png"

    plot(
        [ args.thresholds, args.thresholds ],
        [
            results['false_rejection_rate'][distance_metric_name].values(),
            results['false_acceptance_rate'][distance_metric_name].values()
        ],
        [ "thresholds" ], [ "False Rejection Rate", "False Acceptance Rate" ],
        [ "False Rejection Rate", "False Acceptance Rate" ],
        "thresholds VS. FRR and FAR", show_plot, plot_name,
        "linear", "linear"
    )

# ROC: x = FAR, y = GAR
def compute_ROC_plot(distance_metric_name, show_plot, file_name):

    folder_path = f'''../plots/{distance_metric_name}/{args.threshold_range[0]}_
        {args.threshold_range[1]}_{int(args.threshold_range[2])}'''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plot_name = f"{folder_path}/roc_{file_name}.png"


    plot(
        [ results['false_acceptance_rate'][distance_metric_name].values() ],
        [ results['genuine_acceptance_rate'][distance_metric_name].values() ],
        [ "False Acceptance Rate" ], [ "Genuine Acceptance Rate" ], ["ROC"],
        "ROC", show_plot, plot_name,
        "linear", "linear"
    )

# DET (logarithmic scale): x = FAR, y = FRR
def compute_DET_plot(distance_metric_name, show_plot, file_name):
    folder_path = f'''../plots/{distance_metric_name}/{args.threshold_range[0]}_
        {args.threshold_range[1]}_{int(args.threshold_range[2])}'''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plot_name = f"{folder_path}/det_{file_name}.png"

    plot(
        [results['false_acceptance_rate'][distance_metric_name].values()],
        [results['false_rejection_rate'][distance_metric_name].values()],
        ["False Acceptance Rate"], ["Genuine Acceptance Rate"], ["DET"],
        "DET", show_plot, plot_name,
        # TODO
        # BE WARE! should use "log" "log"
        # BUT current values are too low so log display does not work.
        # "log", "log",
        "linear", "linear"
    )

def compute_plots(show_plot, file_name):
    for metric in distance_metrics:
        compute_threshold_FAR_FRR_plot(metric, show_plot, file_name)

        compute_ROC_plot(metric, show_plot, file_name)

        compute_DET_plot(metric, show_plot, file_name)


results = perform_all_against_all(distance_metrics, args.thresholds, args.verbose)

file_name = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H:%M:%S')

with open("../logs/" + file_name + ".json", "w") as output_log:
    output_log.write(json.dumps(results, indent = 4))
    output_log.close()

with open("../logs/" + file_name + ".err.json", "w") as error_log:
    error_log.write(json.dumps(errors, indent = 4))
    error_log.close()

compute_plots(False, file_name)

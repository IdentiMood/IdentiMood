import argparse
import random
from deepface import DeepFace
import json
import time
from datetime import datetime
import numpy as np

DEFAULT_MODELS_MASK = "10000000"

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
parser.add_argument("-m", "--models-mask", help="Binary mask to decide what Deep Learning models to use for face verification. Mask index meaning: 0 --> VGG, 1 --> OpenFace, 2--> Facenet, 3 --> Facenet512, 4 --> Facebook DeepFace, 5 --> DeepID, 6 --> Dlib, 7 --> ArcFace", type=str)
parser.add_argument("-v", "--verbose", help="Print iteration results", action="store_true")
parser.add_argument("-b", "--begin-at-line", help="Line of input file to begin from", type=int, default = 0)
args = parser.parse_args()

dataset_name = args.input.split('_list_')[1].split('.txt')[0]

models_list = [ 
    'VGG-Face', 'OpenFace', 'Facenet', 'Facenet512', 'DeepFace', 'DeepID',
	'Dlib', 'ArcFace'
]

if args.models_mask == None:
    args.models_mask = DEFAULT_MODELS_MASK
    print("Models mask not provided, defaulting to", DEFAULT_MODELS_MASK)

if not (set(args.models_mask).issubset({'0', '1'}) and bool(args.models_mask)):
    args.models_mask = DEFAULT_MODELS_MASK
    print("Provided models mask not binary, defaulting to", DEFAULT_MODELS_MASK)

if len(args.models_mask) != len(models_list):
    args.models_mask = DEFAULT_MODELS_MASK
    print(
        "Provided models mask not complete, defaulting to", DEFAULT_MODELS_MASK
    )

models_dict = {}

for (i, j) in zip(list(range(0, len(args.models_mask))), models_list):
    if args.models_mask[i] == "1": 
        models_dict[j] = DeepFace.build_model(j)
        print((i, j))


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

# TODO delete it after support for multiple models
# model = DeepFace.build_model('VGG-Face')

def verify(distance_score, threshold):
    return distance_score <= threshold

with open(args.input) as file:
    lines = file.readlines()
    lines = lines[args.begin_at_line : ]
    lines = [line.rstrip() for line in lines]
    if args.shuffle:
        random.shuffle(lines)
    if args.limit is not None:
        lines = lines[ : args.limit]
    total_combinations = (len(lines)**2 - len(lines)) * len(args.thresholds) * \
        len(distance_metrics) * len(models_dict.keys())


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

    # estimated_exec_time_seconds = total_combinations * 0.2
    # print("total combinations: ", total_combinations)
    # print("should take: ", estimated_exec_time_seconds, "seconds")
    # print("should take: ", estimated_exec_time_seconds / 60, "minutes")
    # print("should take: ", estimated_exec_time_seconds / 3600, "hours")
    # print("should take: ", estimated_exec_time_seconds / (3600 * 24), "days")
    # exit()

    current_combination = 1

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

        for model in models_dict.keys():
            genuine_acceptances[metric][model] = dict()
            genuine_rejections[metric][model] = dict()
            false_acceptances[metric][model] = dict()
            false_rejections[metric][model] = dict()

            genuine_attempts[metric][model] = dict()
            impostor_attempts[metric][model] = dict()

            genuine_acceptance_rate[metric][model] = dict()
            genuine_rejection_rate[metric][model] = dict()
            false_acceptance_rate[metric][model] = dict()
            false_rejection_rate[metric][model] = dict()
            error_rate[metric][model] = dict()


            for threshold in thresholds:
                threshold_str = str(threshold)

                genuine_acceptances[metric][model][threshold_str] = 0
                genuine_rejections[metric][model][threshold_str] = 0
                false_acceptances[metric][model][threshold_str] = 0
                false_rejections[metric][model][threshold_str] = 0

                genuine_attempts[metric][model][threshold_str] = 0
                impostor_attempts[metric][model][threshold_str] = 0

                genuine_acceptance_rate[metric][model][threshold_str] = 0
                genuine_rejection_rate[metric][model][threshold_str] = 0
                false_acceptance_rate[metric][model][threshold_str] = 0
                false_rejection_rate[metric][model][threshold_str] = 0
                error_rate[metric][model][threshold_str] = 0

            for first_identity_index in range(0, len(lines)):
                first_identity_name = lines[first_identity_index].split('/')[3]

                for second_identity_index in range(0, len(lines)):

                    if (second_identity_index == first_identity_index): continue

                    try:
                        result = DeepFace.verify(
                            img1_path = lines[first_identity_index],
                            img2_path = lines[second_identity_index],
                            model = models_dict[model]
                        )
                    except ValueError as e:
                        # happens when the img cannot be loaded
                        print(e, lines[first_identity_index], lines[second_identity_index])

                        errors.append({
                            "error": str(e),
                            "img1_path": lines[first_identity_index],
                            "img2_path": lines[second_identity_index],
                        })
                        continue

                    second_identity_name = lines[second_identity_index].split('/')[3]

                    for threshold in thresholds:
                        threshold_str = str(threshold)

                        if (first_identity_name == second_identity_name):
                            genuine_attempts[metric][model][threshold_str] += 1
                        else:
                            impostor_attempts[metric][model][threshold_str] += 1

                        verified = verify(result['distance'], threshold)

                        genuine_acceptances[metric][model][threshold_str] += int(verified and (first_identity_name == second_identity_name))

                        genuine_rejections[metric][model][threshold_str] += int(not verified and not (first_identity_name == second_identity_name))

                        false_acceptances[metric][model][threshold_str] += int(verified and not (first_identity_name == second_identity_name))

                        false_rejections[metric][model][threshold_str] += int(not verified and (first_identity_name == second_identity_name))

                        ga = genuine_attempts[metric][model][threshold_str]

                        if (ga != 0):
                            genuine_acceptance_rate[metric][model][threshold_str] = \
                                genuine_acceptances[metric][model][threshold_str] / ga

                            false_rejection_rate[metric][model][threshold_str] = \
                                false_rejections[metric][model][threshold_str] / ga

                        ia = impostor_attempts[metric][model][threshold_str]

                        if (ia != 0):
                            genuine_rejection_rate[metric][model][threshold_str] = \
                                genuine_rejections[metric][model][threshold_str] / ia

                            false_acceptance_rate[metric][model][threshold_str] = \
                                false_acceptances[metric][model][threshold_str] / ia

                        # should never be the case, but, one can never know... :(
                        if (ga + ia != 0):
                            error_rate[metric][model][threshold_str] = (
                                false_acceptances[metric][model][threshold_str] +
                                false_rejections[metric][model][threshold_str]
                            ) / (ga + ia)

                        if ((current_combination % 4000) == 0):
                            temp_json = {
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

                            print("Storing checkpoint json...")

                            file_name = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S') + "_" + str(current_combination)
                            with open("../logs/identification/checkpoints/" + file_name + ".json", "w") as output_log:
                                output_log.write(json.dumps(temp_json, indent = 2))
                                output_log.close()
                        
                        if (verbose):
                            print(f"Matching faces:  {first_identity_index} ({first_identity_name}) VS {second_identity_index} ({second_identity_name})")
                            print(f"DeepFace says:   {verified}")
                            print(f"Should be:       {first_identity_name == second_identity_name}")
                            print(f"Model:           {model}")
                            print(f"Distance:        {result['distance']}")
                            print(f"Threshold:       {threshold}")
                            print(f"Distance metric: {metric}")
                            print(f"Progress:        {current_combination}/{total_combinations} [{round(current_combination/total_combinations*100, 2)}%]")
                            print("----------------------------------------------------------------------------------------")

                            current_combination += 1


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
        
        for model in models_dict.keys():
            
            for threshold in args.thresholds:
                threshold_str = str(threshold)

                # print("GA[", threshold_str, "]: ", genuine_acceptances[metric][model][threshold_str], genuine_attempts[metric][model][threshold_str])
                # print("GR[", threshold_str, "]: ", genuine_rejections[metric][model][threshold_str], genuine_attempts[metric][model][threshold_str])
                # print("FA[", threshold_str, "]: ", false_acceptances[metric][model][threshold_str], impostor_attempts[metric][model][threshold_str])
                # print("FR[", threshold_str, "]: ", false_rejections[metric][model][threshold_str], impostor_attempts[metric][model][threshold_str])

                print("-----------")

                print(
                    "Genuine attemps[", metric, "][", threshold_str, "]: ",
                    genuine_attempts[metric][model][threshold_str]
                )
                print(
                    "Impostor attemps[", metric, "][", threshold_str, "]: ",
                    impostor_attempts[metric][model][threshold_str]
                )
                print()
                print(
                    "GAR[", metric, "][", threshold_str, "]: ",
                    genuine_acceptance_rate[metric][model][threshold_str]
                )
                print(
                    "GRR[", metric, "][", threshold_str, "]: ",
                    genuine_rejection_rate[metric][model][threshold_str]
                )
                print(
                    "FAR[", metric, "][", threshold_str, "]: ",
                    false_acceptance_rate[metric][model][threshold_str]
                )
                print(
                    "FRR[", metric, "][", threshold_str, "]: ",
                    false_rejection_rate[metric][model][threshold_str]
                )
                print(
                    "Error rate[", metric, "][", threshold_str, "]: ", error_rate[metric][model][threshold_str]
                )
                print()

start_time = time.time()
results = perform_all_against_all(distance_metrics, args.thresholds, args.verbose)
end_time = time.time()
print("Total execution time (s)       : ", end_time - start_time)
print("Average single match execution time (s): ", (end_time - start_time) / total_combinations / len(args.thresholds))

file_name = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')

results["dataset_name"] = dataset_name

with open("../logs/identification/" + file_name + ".json", "w") as output_log:
    output_log.write(json.dumps(results, indent = 4))
    output_log.close()

with open("../logs/identification/" + file_name + ".err.json", "w") as error_log:
    error_log.write(json.dumps(errors, indent = 4))
    error_log.close()

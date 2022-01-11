from plotter import plot
import json
import sys
import time
from datetime import datetime
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--show-plot", 
    help = "Whether to show the plots as they are computed", action="store_true"
)
parser.add_argument(
    "-d", "--plot-save-dir", help = "Directory in which to store computed plots"
)
parser.add_argument(
    "-i", "--input-json", help = "JSON storing data to compute plots from"
)

args = parser.parse_args()

if args.input_json == None:
    print("ERROR: Must specify a JSON file path.")
    exit(-1)

if args.plot_save_dir == None:
    args.plot_save_dir = "./plots"

for sub_dir in ["/threshold_vs_frr_far", "/roc", "/det"]:
    if not os.path.exists(args.plot_save_dir + sub_dir):
        os.makedirs(args.plot_save_dir + sub_dir)

# Opening JSON file
f = open(args.input_json)

# returns JSON object as a dictionary
json_content = json.load(f)

# Closing file
f.close()

print(args.input_json)

try:
    dataset_name = json_content["dataset_name"]
except KeyError as e:
    print("Can't find dataset name, exiting...")
    exit(-1)

if dataset_name == "ExtendedYaleB_accepted_by_deepface":
    print("Invalid dataset, exiting...")
    exit(-1)

# beautify dataset names
dataset_name = dataset_name.replace("_balanced", " (balanced)")
dataset_name = dataset_name.replace("_accepted_by_deepface", "")

metrics = list(json_content["genuine_acceptances"])

models_temp_key = list(json_content["genuine_acceptances"])[0]
models = list(json_content["genuine_acceptances"][models_temp_key])

time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')

# plotting:
# threshold vs. FAR & FRR --> x = thresholds, y = FRR, FAR
# ROC                     --> x = FAR, y = GAR
# DET (logarithmic scale) --> x = FAR, y = FRR
for metric in metrics:
    for model in models: 

        # it does NOT make any sense to evaluate a model with its training set
        if "VGG-Face" in model and "VGG-Face" in dataset_name:
            print("SKIPPING", model, dataset_name, "combo")
            continue

        far = list(json_content["false_acceptance_rate"][metric][model].values())
        far_np = np.array(far)
        far_max_val = far_np.max()
        far_max_ind = far_np.argmax()

        frr = list(json_content["false_rejection_rate"][metric][model].values())
        frr_np = np.array(frr)
        frr_min_val = frr_np.min()
        frr_min_ind = frr_np.argmin()

        far_frr_ind = max(far_max_ind, frr_min_ind) + 1

        thresholds = list(
            json_content["false_acceptance_rate"][metric][model].keys()
        )
        thresholds_np = np.array(thresholds).astype(np.float64)[:far_frr_ind]

        plot_file_full_path = args.plot_save_dir + "/threshold_vs_frr_far/" + \
            metric + "_" + model + "_" + time_stamp + ".png"
        
        # plotting:
        # threshold vs. FAR & FRR --> x = thresholds, y = FRR, FAR
        plot(
            # x_axis = [thresholds, thresholds],
            x_axis = [thresholds_np, thresholds_np],
            y_axis = [far[:far_frr_ind], frr[:far_frr_ind]],
            x_label = ["thresholds"], y_label = ["FRR (lower is better)", "FAR (lower is better)"],
            line_label = [ "False Acceptance Rate", "False Rejection Rate" ],
            plot_name = f"thresholds VS. FRR & FAR\nDataset: {dataset_name}\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            x_axis_scale = "linear", y_axis_scale = "linear",
            legend_font_size = "small"
        )

        gar = list(json_content["genuine_acceptance_rate"][metric][model].values())
        gar_np = np.array(gar)
        gar_max_val = gar_np.max()
        gar_max_ind = gar_np.argmax()

        gar_far_ind = max(gar_max_ind, far_max_ind) + 1

        plot_file_full_path = args.plot_save_dir + "/roc/" + metric + "_" + \
            model + "_" + time_stamp + ".png"
        
        # plotting:
        # ROC --> x = FAR, y = GAR
        plot(
            x_axis = [far[:gar_far_ind]],
            y_axis = [gar[: gar_far_ind]],
            x_label = ["FAR (lower is better)"], y_label = ["GAR (higher is better)"],
            line_label = ["ROC"],
            plot_name = f"ROC\nDataset: {dataset_name}\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            x_axis_scale = "linear", y_axis_scale = "linear",
            legend_font_size = "medium"
        )

        plot_file_full_path = args.plot_save_dir + "/det/" + metric + "_" + \
            model + "_" + time_stamp + ".png"
        
        # plotting:
        # DET (logarithmic scale) --> x = FAR, y = FRR
        plot(
            x_axis = [far[:far_frr_ind]],
            y_axis = [frr[:far_frr_ind]],
            x_label = ["FAR (lower is better)"], y_label = ["FRR (lower is better)"],
            line_label = ["DET"],
            plot_name = f"DET\nDataset: {dataset_name}\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            # TODO DET should take logarithmic scale for both axes
            # find a way to set them
            x_axis_scale = "linear", y_axis_scale = "linear",
            legend_font_size = "medium"
        )

from plotter import plot
import json
import sys
import time
from datetime import datetime
import os
import argparse

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

print(args)

# Opening JSON file
f = open(args.input_json)

# returns JSON object as a dictionary
json_content = json.load(f)

# Closing file
f.close()

metrics = list(json_content["genuine_acceptances"])

models_temp_key = list(json_content["genuine_acceptances"])[0]
models = list(json_content["genuine_acceptances"][models_temp_key])

thresholds = list(json_content["genuine_acceptances"][metrics[0]][models[0]])

time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')

# plotting:
# threshold vs. FAR & FRR --> x = thresholds, y = FRR, FAR
# ROC                     --> x = FAR, y = GAR
# DET (logarithmic scale) --> x = FAR, y = FRR
for metric in metrics:
    for model in models: 
        far = list(json_content["false_acceptance_rate"][metric][model].values())

        frr = list(json_content["false_rejection_rate"][metric][model].values())

        plot_file_full_path = args.plot_save_dir + "/threshold_vs_frr_far/" + \
            metric + "_" + model + "_" + time_stamp + ".png"

        # plotting:
        # threshold vs. FAR & FRR --> x = thresholds, y = FRR, FAR
        plot(
            x_axis = [far, frr],
            y_axis = [thresholds, thresholds],
            x_label = "thresholds", y_label = "FRR, FAR",
            line_label = [ "False Rejection Rate", "False Acceptance Rate" ],
            plot_name = f"thresholds VS. FRR & FAR\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            x_axis_scale = "linear", y_axis_scale = "linear"
        )

        gar = list(json_content["genuine_acceptance_rate"][metric][model].values())

        plot_file_full_path = args.plot_save_dir + "/roc/" + metric + "_" + \
            model + "_" + time_stamp + ".png"
        
        # plotting:
        # ROC --> x = FAR, y = GAR
        plot(
            x_axis = [far],
            y_axis = [gar],
            x_label = "FAR", y_label = "GAR",
            line_label = ["ROC"],
            plot_name = f"ROC\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            x_axis_scale = "linear", y_axis_scale = "linear"
        )

        plot_file_full_path = args.plot_save_dir + "/det/" + metric + "_" + \
            model + "_" + time_stamp + ".png"
        
        # plotting:
        # DET (logarithmic scale) --> x = FAR, y = FRR
        plot(
            x_axis = [far],
            y_axis = [frr],
            x_label = "FAR", y_label = "FRR",
            line_label = ["DET"],
            plot_name = f"DET\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = plot_file_full_path,
            # TODO DET should take logarithmic scale for both axes
            # find a way to set them
            x_axis_scale = "linear", y_axis_scale = "linear"
        )

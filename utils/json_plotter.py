from plotter import plot
import json
import sys
import time
from datetime import datetime
import os
import argparse
import numpy as np

def create_plot_path(dataset_name, model, metric, plot_name):
    time_stamp = datetime.fromtimestamp(
        time.time()).strftime('%y_%m_%d_%H-%M-%S'
    )

    file_path = args.plot_save_dir + "/" + dataset_name + "/" + \
            model + "/" + metric + "/"

    if not os.path.exists(file_path):
        os.makedirs(file_path)
            
    file_name = plot_name + "_" + time_stamp + ".png"

    return file_path + file_name

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

model_color_map = {
    'VGG-Face' : "red", 
    'OpenFace' : "lime", 
    'Facenet' : "blue", 
    'Facenet512' : "purple", 
    'DeepFace' : "coral", 
    'DeepID' : "olive",
	'Dlib' : "turquoise", 
    'ArcFace' : "cornflowerblue"
}

model_color_map_alt = {
    'VGG-Face' : "maroon", 
    'OpenFace' : "darkgreen", 
    'Facenet' : "midnightblue", 
    'Facenet512' : "blueviolet", 
    'DeepFace' : "orangered", 
    'DeepID' : "darkolivegreen",
	'Dlib' : "teal", 
    'ArcFace' : "dodgerblue"
}

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
        frr = list(json_content["false_rejection_rate"][metric][model].values())

        if metric == "cosine":
            far_frr_ind = len(far)
        else:
            far_np = np.array(far)
            far_max_val = far_np.max()
            far_max_ind = far_np.argmax()

            frr_np = np.array(frr)
            frr_min_val = frr_np.min()
            frr_min_ind = frr_np.argmin()

            far_frr_ind = max(far_max_ind, frr_min_ind) + 1

        thresholds = list(
            json_content["false_acceptance_rate"][metric][model].keys()
        )
        thresholds_np = np.array(thresholds).astype(np.float64)[:far_frr_ind]
        
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
            plot_file_full_path = create_plot_path(dataset_name, model, metric, "thresholds_vs_FAR_FRR"),
            x_axis_scale = "linear", y_axis_scale = "linear",
            legend_font_size = "small",
            color = [model_color_map[model], model_color_map_alt[model]]
        )

        gar = list(json_content["genuine_acceptance_rate"][metric][model].values())
        gar_np = np.array(gar)
        gar_max_val = gar_np.max()
        gar_max_ind = gar_np.argmax()

        if metric == "cosine":
            gar_far_ind = len(gar)
        else:   
            gar_far_ind = max(gar_max_ind, far_max_ind) + 1
        
        # plotting:
        # ROC --> x = FAR, y = GAR
        plot(
            x_axis = [far[:gar_far_ind]],
            y_axis = [gar[: gar_far_ind]],
            x_label = ["FAR (lower is better)"], y_label = ["GAR (higher is better)"],
            line_label = ["ROC"],
            plot_name = f"ROC\nDataset: {dataset_name}\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = create_plot_path(dataset_name, model, metric, "ROC"),
            x_axis_scale = "linear", y_axis_scale = "linear",
            legend_font_size = "medium",
            color = [model_color_map[model]]
        )
        
        # plotting:
        # DET (logarithmic scale) --> x = FAR, y = FRR
        plot(
            x_axis = [far[:far_frr_ind]],
            y_axis = [frr[:far_frr_ind]],
            x_label = ["FAR (higher is better)"], y_label = ["FRR (higher is better)"],
            line_label = ["DET"],
            plot_name = f"DET\nDataset: {dataset_name}\nDistance metric: {metric}. Deep Learning model: {model}",
            show_plot = args.show_plot,
            plot_file_full_path = create_plot_path(dataset_name, model, metric, "DET"),
            x_axis_scale = "log", y_axis_scale = "log",
            legend_font_size = "medium",
            color = [model_color_map[model]]
        )
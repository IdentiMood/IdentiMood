from plotter import plot
import json
import argparse
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt


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
    args.plot_save_dir = "./plots/emotion"


if not os.path.exists(args.plot_save_dir):
    os.makedirs(args.plot_save_dir)

# Opening JSON file
f = open(args.input_json)

# returns JSON object as a dictionary
json_content = json.load(f)

# Closing file
f.close()

time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')

models = list(json_content.keys())

thresholds = list(json_content[models[0]].keys())
# needed by the amazing MatPlotLib, in order to show readable x ticks :)
thresholds_rounded = [round(float(threshold), 2) for threshold in thresholds]

x_axis = [thresholds_rounded for model in models]

y_axis = dict(list())

for model in models:
    y_axis[model] = list()

    for threshold in thresholds:
        
        num_correct = json_content[model][threshold]["correct"]
        num_wrong = json_content[model][threshold]["wrong"]
        num_tot = num_correct + num_wrong

        positive_ratio = num_correct / num_tot * 100
        
        y_axis[model].append(positive_ratio)

line_label = [model for model in models]

plot_file_full_path = args.plot_save_dir  + "_" + time_stamp + ".png"

for (x, y, l) in zip(x_axis, [y_axis[model] for model in models], line_label):
    plt.plot(x, y, label = l)

plt.xlabel("thresholds")
plt.ylabel("% of correct emotion prediction")

plt.legend()

plt.show()
plt.draw()

plt.clf()


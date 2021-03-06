import json
import argparse
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--show-plot",
    help="Whether to show the plots as they are computed",
    action="store_true",
)
parser.add_argument(
    "-d", "--plot-save-dir", help="Directory in which to store computed plots"
)
parser.add_argument(
    "-i", "--input-json", help="JSON storing data to compute plots from"
)

args = parser.parse_args()

if args.input_json == None:
    print("ERROR: Must specify a JSON file path.")
    exit(-1)

if args.plot_save_dir == None:
    args.plot_save_dir = "./plots/emotion/"


if not os.path.exists(args.plot_save_dir):
    os.makedirs(args.plot_save_dir)

# Opening JSON file
f = open(args.input_json)

# returns JSON object as a dictionary
json_content = json.load(f)

# Closing file
f.close()

time_stamp = datetime.fromtimestamp(time.time()).strftime("%y_%m_%d_%H-%M-%S")

models = list(json_content.keys())

thresholds = list(json_content[models[0]].keys())
# needed by the amazing MatPlotLib, in order to show readable x ticks :)
thresholds_rounded = [round(float(threshold), 2) for threshold in thresholds]

x_axis = [thresholds_rounded for model in models]

y_axis_correct = dict(list())
y_axis_wrong = dict(list())

for model in models:

    y_axis_correct[model] = list()
    y_axis_wrong[model] = list()

    for threshold in thresholds:

        num_correct = json_content[model][threshold]["correct"]
        num_wrong = json_content[model][threshold]["wrong"]
        num_tot = num_correct + num_wrong

        positive_ratio = num_correct / num_tot * 100

        # y_axis_correct[model].append(num_correct)
        # y_axis_wrong[model].append(num_wrong)
        y_axis_correct[model].append(num_correct / num_tot * 100)
        y_axis_wrong[model].append(num_wrong / num_tot * 100)

line_label_correct = [
    model + " (correct emotion verification)" for model in models
]
line_label_wrong = [
    model + " (wrong emotion verification)" for model in models
]

plot_file_full_path = args.plot_save_dir + time_stamp + ".png"

fig, ax1 = plt.subplots()

ax1.set_xlabel(
    "threshold (as delta between the first two probability scores)", 
    fontweight='bold',
    fontsize = 16
)
ax1.set_ylabel(
    "% of correct emotion idenfications (higher is better)",
    fontweight='bold',
    fontsize = 16
)

ax2 = ax1.twinx()
ax2.set_ylabel(
    "% of wrong emotion idenfications (lower is better)",
    fontweight='bold',
    fontsize = 16
)

fig.suptitle(
    "thresholds VS. correct & wrong emotion verification ratios",
    fontweight='bold',
    fontsize = 16
)
plt.title(
    "Datasets: TUTFS, KDEF, yalefaces & VGG-Face2",
    fontweight='bold',
    fontsize = 14
)

# color_correct = [
#     "#9DA1AA", "#1C1C1C", "#89AC76", "#8B8C7A", "#CC0605", "#AF2B1E", "#F44611",
#     "#2E3A23"
# ]

# color_wrong = [
#     "#015D52", "#00BB2D", "#C7B446", "#8E402A", "#2F4538", "#6D6552", "#49678D",
#     "#6C6874"
# ]

color_correct = [
    "g",
    "g",
    "g",
    "g",
    "g",
    "g",
    "g",
    "g",
]

color_wrong = [
    "r",
    "r",
    "r",
    "r",
    "r",
    "r",
    "r",
    "r",
]

for (
    x_axis,
    y_correct,
    y_wrong,
    label_correct,
    label_wrong,
    color_correct,
    color_wrong,
) in zip(
    x_axis,
    [y_axis_correct[model] for model in models],
    [y_axis_wrong[model] for model in models],
    line_label_correct,
    line_label_wrong,
    color_correct,
    color_wrong,
):
    
    ax1.plot(x_axis, y_correct, label=label_correct, color=color_correct)
    # ax1.legend()
    ax1.legend(
        loc="upper left", 
        ncol=1, 
        bbox_to_anchor=(0.075, 1), 
        fancybox=True, 
        shadow=False,
        prop = {'size': 10, "weight" : "bold"}
    )

    # draw an invisible point to normalize the vertical axis ticks
    ax1.plot([0], [100])

    ax2.plot(x_axis, y_wrong, label=label_wrong, color=color_wrong)
    # ax2.legend()
    ax2.legend(
        loc="upper right",
        ncol=1,
        bbox_to_anchor=(0.925, 1),
        fancybox=True,
        shadow=False,
        prop = {'size': 10, "weight" : "bold"}
    )

intersection_model = "DeepEmotion"
l1 = list(zip(x_axis, y_axis_correct[intersection_model]))
l2 = list(zip(x_axis, y_axis_wrong[intersection_model]))

line1 = LineString(l1)
line2 = LineString(l2)

intersection = line1.intersection(line2)

eer_vertical_line_y = np.linspace(0, 100, 100)
eer_vertical_line_x = np.empty(100)
eer_vertical_line_x.fill(intersection.x)

eer_threshold = round(intersection.x, 2)
custom_ticks = np.append(ax1.get_xticks(), eer_threshold)
sorted_custom_ticks = np.sort(custom_ticks)

ax1.set_xticks(sorted_custom_ticks)
ax1.tick_params(axis="x", labelrotation=45)
plt.tight_layout()

plt.plot(
    eer_vertical_line_x,
    eer_vertical_line_y,
    linestyle="-.",
    color="blue",
)


if plot_file_full_path:
    plt.savefig(plot_file_full_path, dpi=300)

if args.show_plot:
    plt.show()
    plt.draw()

plt.clf()

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
    args.plot_save_dir = "./plots/emotion/"


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

line_label_correct = [model + " (correct emotion recognition)" for model in models]
line_label_wrong = [model + " (wrong emotion recognition)" for model in models]

plot_file_full_path = args.plot_save_dir + time_stamp + ".png"

fig, ax1 = plt.subplots()

ax1.set_xlabel("thresholds")
ax1.set_ylabel("% of correct emotion idenfications (higher is better)")

ax2 = ax1.twinx()
ax2.set_ylabel("% of wrong emotion idenfications (lower is better)")

fig.suptitle(
    "thresholds VS. correct & wrong emotion recognition ratios"
)
plt.title("Dataset: TUTFS x KDEF x yalefaces")

color_correct = [
    "#9DA1AA", "#1C1C1C", "#89AC76", "#8B8C7A", "#CC0605", "#AF2B1E", "#F44611", 
    "#2E3A23"
]

color_wrong = [
    "#015D52", "#00BB2D", "#C7B446", "#8E402A", "#2F4538", "#6D6552", "#49678D",
    "#6C6874"
]

for (x_axis, y_correct, y_wrong, label_correct, label_wrong, color_correct, color_wrong) in zip(x_axis, [y_axis_correct[model] for model in models], [y_axis_wrong[model] for model in models], line_label_correct, line_label_wrong, color_correct, color_wrong):

    ax1.plot(x_axis, y_correct, label = label_correct, color = color_correct)
    # ax1.legend()
    ax1.legend(
        loc='upper left', 
        ncol=1,
        bbox_to_anchor=(0.075, 1),
        fancybox=True, 
        shadow=False
    )
    
    ax2.plot(x_axis, y_wrong, label = label_wrong, color = color_wrong)
    # ax2.legend()
    ax2.legend(
        loc='upper right', 
        ncol=1, 
        bbox_to_anchor=(0.925, 1),
        fancybox=True, 
        shadow=False
    )

if (plot_file_full_path): 
    plt.savefig(plot_file_full_path, dpi = 300)

if (args.show_plot):
        plt.show()
        plt.draw()

plt.clf()


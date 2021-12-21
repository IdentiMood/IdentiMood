import matplotlib.pyplot as plt
import os


def plot(
    x_axis, y_axis,
    x_label = "x - axis", y_label = "y - label", line_label = [],
    plot_name = " ", plot = True,
    plot_file_name_with_path = None,
    x_axis_scale = "linear", y_axis_scale = "linear",
    ticks_to_use = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
):
    # naming the x axis
    plt.xlabel(x_label)
    # naming the y axis
    plt.ylabel(y_label)

    # giving a title to my graph
    plt.title(plot_name)

    plt.xscale(x_axis_scale)
    plt.yscale(y_axis_scale)

    if (ticks_to_use != None):
        plt.xticks(ticks_to_use)
        plt.yticks(ticks_to_use)

    # TODO use zip
    #      loop over zipped x_axis and y_axis
    #      in theory, as it is now it's ok BUT it doesn't have error handling that zip would
    #      provide
    for i in range(0, len(y_axis)):
        plt.plot(x_axis[i], y_axis[i], label = line_label[i])

    plt.legend()

    if (plot_file_name_with_path): plt.savefig(plot_file_name_with_path)

    # function to show the plot
    if (plot):
        plt.show()
        plt.draw()

    plt.clf()

# TODO finish to code this
# take all needed params from outside!
def __compute_threshold_FAR_FRR_plot(x_axis_list, y_axis_list):
    plot(
        x_axis_list,
        y_axis_list,
        [ "thresholds" ], [ "False Rejection Rate", "False Acceptance Rate" ],
        [ "False Rejection Rate", "False Acceptance Rate" ],
        "thresholds VS. FRR and FAR", show_plot, plot_name,
        "linear", "linear"
    )

# TFF: x = thresholds, y = FRR, FAR
def compute_threshold_FAR_FRR_plot(distance_metric_name, show_plot, file_name, results):

    folder_path = f"../plots/{distance_metric_name}/{thresholds[0]}_" + \
        f"{thresholds[-1]}_{int(len(thresholds))}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plot_name = f"{folder_path}/tff_{file_name}.png"

    plot(
        [ thresholds, thresholds ],
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
def compute_ROC_plot(distance_metric_name, show_plot, file_name, results):

    folder_path = f"../plots/{distance_metric_name}/{thresholds[0]}_" + \
        f"{thresholds[-1]}_{int(len(thresholds))}"
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
def compute_DET_plot(distance_metric_name, show_plot, file_name, results):
    folder_path = f"../plots/{distance_metric_name}/{thresholds[0]}_" + \
        f"{thresholds[-1]}_{int(len(thresholds))}"
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
        compute_threshold_FAR_FRR_plot(metric, show_plot, file_name, results)

        compute_ROC_plot(metric, show_plot, file_name, results)

        compute_DET_plot(metric, show_plot, file_name, results)

# compute_plots(bool(int(sys.argv[2])), "TODO add plot name")
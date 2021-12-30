from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedFormatter, FixedLocator, MultipleLocator
import numpy as np


def plot(
    x_axis = [None], y_axis = [None],
    x_label = ["x - axis"], y_label = ["y - label"], 
    line_label = [],
    plot_name = " ", 
    show_plot = True,
    plot_file_full_path = None,
    x_axis_scale = "linear", y_axis_scale = "linear"
):

    plt.title(plot_name)

    plt.xscale(x_axis_scale)
    plt.yscale(y_axis_scale)

    if (len(y_axis) == 2):
        _, ax1 = plt.subplots()
        ax1.set_xlabel(x_label[0])
        ax1.set_ylabel(y_label[0])
        # so as the plotted line does NOT overlap the axis
        ax1.set_ylim(bottom = -0.005)

        ax2 = ax1.twinx()
        ax2.set_ylabel(y_label[1])
        # so as the plotted line does NOT overlap the axis
        ax2.set_ylim(bottom = -0.005)

    else:
        plt.xlabel(x_label[0])
        plt.ylabel(y_label[0])

    for (x, y, l) in zip(x_axis, y_axis, line_label):
        plt.plot(x, y, label = l)

    plt.legend()

    if (plot_file_full_path): plt.savefig(plot_file_full_path)

    if (show_plot):
        plt.show()
        plt.draw()

    plt.clf()

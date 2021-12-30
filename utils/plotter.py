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
    x_axis_scale = "linear", y_axis_scale = "linear",
    ticks_to_use = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
):

    # giving a title to my graph
    plt.title(plot_name)

    # setting scale of x axis
    plt.xscale(x_axis_scale)
    # setting scale of y axis
    plt.yscale(y_axis_scale)

    if (len(y_axis) == 2):
        _, ax1 = plt.subplots()
        ax1.set_xlabel(x_label[0])
        ax1.set_ylabel(y_label[0])

        # ax1.xaxis.set_major_locator(ticker.LinearLocator())
        # ax1.xaxis.set_minor_locator(ticker.MultipleLocator())


        # ax1.xaxis.set_minor_locator(ticker.NullLocator())

        ## Works, BUT no idea why...
        # ax1.xaxis.set_major_locator(MaxNLocator(12))
        # ax1.xaxis.set_major_formatter(FixedFormatter(ticks_to_use))
        ##

        ax2 = ax1.twinx()
        ax2.set_ylabel(y_label[1])

        # Use any of those to handle graphs in which line is exactly 
        # on the axis.
        # This way the axis is shifted down and the line gets back to a visible
        # state

        # solution 1
        # ax1.set_ymargin(0.5)
        # ax2.set_ymargin(0.5)

        # solution 2
        ax1.set_ylim(bottom = -0.01)
        ax2.set_ylim(bottom = -0.01)


    else:
        # naming the x axis
        plt.xlabel(x_label[0])
        # naming the y axis
        plt.ylabel(y_label[0])
    
        plt.xticks(ticks_to_use)
        plt.yticks(ticks_to_use)

    for (x, y, l) in zip(x_axis, y_axis, line_label):
        plt.plot(x, y, label = l)

    plt.legend()

    if (plot_file_full_path): plt.savefig(plot_file_full_path)

    # function to show the plot
    if (show_plot):
        plt.show()
        plt.draw()

    plt.clf()

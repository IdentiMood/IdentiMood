import matplotlib.pyplot as plt
import os


def plot(
    x_axis = None, y_axis = None,
    x_label = "x - axis", y_label = "y - label", 
    line_label = [],
    plot_name = " ", 
    show_plot = True,
    plot_file_full_path = None,
    x_axis_scale = "linear", y_axis_scale = "linear",
    ticks_to_use = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
):
    # naming the x axis
    plt.xlabel(x_label)
    # naming the y axis
    plt.ylabel(y_label)

    # giving a title to my graph
    plt.title(plot_name)

    # setting scale of x axis
    plt.xscale(x_axis_scale)

    # setting scale of y axis
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

    if (plot_file_full_path): plt.savefig(plot_file_full_path)

    # function to show the plot
    if (show_plot):
        plt.show()
        plt.draw()

    plt.clf()

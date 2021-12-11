import matplotlib.pyplot as plt

def plot(
    x_axis, y_axis,
    x_label = "x - axis", y_label = "y - label", plot_name = " ", plot = True,
    plot_file_name_with_path = None,
    x_axis_scale = "linear", y_axis_scale = "linear"
):
    # naming the x axis
    plt.xlabel(x_label)
    # naming the y axis
    plt.ylabel(y_label)

    # giving a title to my graph
    plt.title(plot_name)

    plt.set_xscale(x_axis_scale)
    plt.set_yscale(y_axis_scale)

    # TODO use zip
    #      loop over zipped x_axis and y_axis
    #      in theory, as it is now it's ok BUT it doesn't have error handling that zip would
    #      provide
    for i in range(0, len(y_axis)):
        plt.plot(x_axis[i], y_axis[i], label = y_label[i])

    plt.legend()

    if (plot_file_name_with_path): plt.savefig(plot_file_name_with_path)

    # function to show the plot
    if (plot):
        plt.show()
        plt.draw()

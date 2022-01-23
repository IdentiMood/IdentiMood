from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString


def plot(
    x_axis = [None], y_axis = [None],
    x_label = ["x - axis"], y_label = ["y - label"], 
    line_label = [],
    plot_name = " ", 
    show_plot = True,
    plot_file_full_path = None,
    x_axis_scale = "linear", y_axis_scale = "linear",
    legend_font_size = "medium",
    color = ["black"]
):

    plt.rc('legend', fontsize = legend_font_size)

    bool_plot_intersection = False

    plt.xscale(x_axis_scale)
    plt.yscale(y_axis_scale)

    if (len(y_axis) == 2):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(x_label[0])
        ax1.set_ylabel(y_label[0])
        # so as the plotted line does NOT overlap the axis
        ax1.set_ylim(bottom = -0.005)
        
        fig.suptitle(plot_name, fontsize = 8)

        ax2 = ax1.twinx()
        ax2.set_ylabel(y_label[1])
        # so as the plotted line does NOT overlap the axis
        ax2.set_ylim(bottom = -0.005)

        l1 = list(zip(x_axis[0], y_axis[0]))
        l2 = list(zip(x_axis[0], y_axis[1]))

        line1 = LineString(l1)
        line2 = LineString(l2)

        intersection = line1.intersection(line2)
        
        eer_vertical_line_y = np.linspace(0, intersection.y, 100)
        eer_vertical_line_x = np.empty(100)
        eer_vertical_line_x.fill(intersection.x)

        eer_threshold = (round(intersection.x, 2))
        custom_ticks = np.append(ax1.get_xticks(), eer_threshold)
        sorted_custom_ticks = np.sort(custom_ticks)

        ax1.set_xticks(sorted_custom_ticks)
        ax1.tick_params(axis = 'x',labelrotation = 45)
        plt.tight_layout()
                
        bool_plot_intersection = True

    else:
        plt.suptitle(plot_name, fontsize = 8)
        
        plt.xlabel(x_label[0])
        plt.ylabel(y_label[0])

    for (i, x, y, l) in zip(range(0, len(y_axis)), x_axis, y_axis, line_label):
        plt.plot(x, y, label = l, color = color[i])

    if bool_plot_intersection:
        
        plt.plot(
            eer_vertical_line_x, eer_vertical_line_y,
            linestyle='dashed', color = "magenta",
            label = "Threshold to get EER"
        )
        
        plt.plot(
            intersection.x, intersection.y, marker = "o", markersize = 6, 
            markeredgecolor = "black", markerfacecolor = "black", 
            color = "black", label = "Equal Error Rate"
        )

    plt.legend()

    if (plot_file_full_path): plt.savefig(plot_file_full_path, dpi = 300)

    if (show_plot):
        plt.show()
        plt.draw()

    plt.clf()
    plt.close()

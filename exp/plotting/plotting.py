"""Collection of plotting commands for plotting mean/average of runs."""

from itertools import cycle

import matplotlib
import pandas
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

matplotlib.use("Agg")


class OptimizationPlot:
    """Collection of plotting commands for optimization plots."""

    @staticmethod
    def plot_metric(
        csv_file,
        plot_std=True,
        std_alpha=0.5,
        scale_steps=1,
        label=None,
        linestyle=None,
    ):
        """Add plot of a metric.

        Parameters:
        -----------
        csv_file : (str)
            Path to the .csv file, require columns 'step', 'mean'and 'std'
        plot_std : (bool)
            Add shaded region one standard deviation around the mean
        scale_step : (float)
            Scale the steps (x-axis) by a ratio (e.g. training set size)
        alpha : (float) between 0 and 1
            Transparency of the standard deviation shade plot
        label : (str)
            Label of the plot, no label if left None
        linestyle : (str)
            Line style for mean value, `'-'`, `'--'`, `'-.'`, or `':'`.
            Default: `'-'`
        """
        step, mean, std = OptimizationPlot.read_csv(csv_file, scale_steps=scale_steps)
        OptimizationPlot.plot_mean(step, mean, label=label, linestyle=linestyle)
        if plot_std:
            OptimizationPlot.plot_std(step, mean, std, alpha=std_alpha)

    @staticmethod
    def plot_mean(steps, mean, label=None, linestyle=None):
        """Plot the mean value."""
        linestyle = "-" if linestyle is None else linestyle
        plt.plot(steps, mean, label=label, linestyle=linestyle)

    @staticmethod
    def plot_std(steps, mean, std, alpha=0.5):
        """Plot sigma-interval around the mean."""
        # spline_lower = InterpolatedUnivariateSpline(steps, mean - std, k=3)
        # spline_upper = InterpolatedUnivariateSpline(steps, mean + std, k=3)
        # steps_fine = np.linspace(np.min(steps),
        #                         np.max(steps),
        #                         5*len(steps))

        # plt.fill_between(steps_fine,
        #                 spline_lower(steps_fine),
        #                 spline_upper(steps_fine),
        #                 alpha=alpha)
        plt.fill_between(steps, mean - std, mean + std, alpha=alpha)

    @staticmethod
    def read_csv(csv_file, scale_steps=1):
        """Read CSV summmary file, return step, mean, std."""
        data = pandas.read_csv(csv_file)
        step = data["step"] / scale_steps
        mean = data["mean"]
        std = data["std"]
        return step, mean, std

    @staticmethod
    def save_as_tikz(out_file, pdf_preview=True):
        """Save TikZ figure using matplotlib2tikz. Optional PDF out."""
        tex_file, pdf_file = [
            "{}.{}".format(out_file, extension) for extension in ["tex", "pdf"]
        ]
        tikz_save(
            tex_file,
            override_externals=True,
            tex_relative_path_to_data="../../fig/",
            extra_axis_parameters={"mystyle"},
        )
        if pdf_preview:
            plt.savefig(pdf_file, bbox_inches="tight")

    @staticmethod
    def create_standard_plot(
        xlabel, ylabel, csv_files, labels, scale_steps=1, plot_std=True, std_alpha=0.3
    ):
        """Standard plot of the same metric for different optimizers."""
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for label, csv, linestyle in zip(
            labels, csv_files, OptimizationPlot.linestyles()
        ):
            OptimizationPlot.plot_metric(
                csv,
                plot_std=plot_std,
                std_alpha=std_alpha,
                scale_steps=scale_steps,
                label=label,
                linestyle=linestyle,
            )

    @staticmethod
    def linestyles():
        """Cycle through all different linestyles of `matplotlib`."""
        _linestyles = ["-", "--", "-.", ":"]
        for style in cycle(_linestyles):
            yield style

    @staticmethod
    def post_process(tikz_file):
        """Remove from matplotlib2tikz export what should be configurable.

        Write processed file to `tikz_file + '_processed.tex'`.
        """
        with open(tikz_file + ".tex", "r") as f:
            content = f.readlines()

        # remove lines containing these specifications
        to_remove = [
            r"x grid style",
            r"y grid style",
            r"tick align",
            r"\addlegendimage",
            r"legend cell align",
            r"legend style",
            r"tick pos",
            r"xmin",
            r"xmax",
            r"ymin",
            r"ymax",
            r"ymajorticks",
            r"xmajorticks",
            r"axis line style",
        ]

        for pattern in to_remove:
            content = [c for c in content if pattern not in c]

        content = "".join(content)

        # remove line width specifications
        linewidths = [
            r"ultra thick",
            r"very thick",
            r"semithick",
            r"thick",
            r"very thin",
            r"ultra thin",
            r"thin",
        ]

        for width in linewidths:
            content = content.replace(width, "")

        out_file = tikz_file + "_processed.tex"
        with open(out_file, "w") as f:
            f.write(content)


class WallPlot(OptimizationPlot):
    @staticmethod
    def read_csv(csv_file):
        """Read CSV summmary file, return wall, value"""
        data = pandas.read_csv(csv_file)
        wall = data["wall"]
        value = data["value"]
        return wall, value

    @staticmethod
    def create_standard_plot(
        xlabel, ylabel, csv_files, labels, return_shortest_wall=True, forget=None
    ):
        """Standard wall-value plot of the same metric."""
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        shortest_wall = float("inf")

        if forget is None:
            forget = []

        for plot_idx, (label, csv_list, linestyle) in enumerate(
            zip(labels, csv_files, WallPlot.linestyles())
        ):
            for idx, csv in enumerate(csv_list):
                used_label = label if idx == 0 else None

                if plot_idx in forget:
                    wall, value = [], []
                    used_label = None
                else:
                    wall, value = WallPlot.read_csv(csv)
                    wall, value = list(wall), list(value)

                    if wall[-1] < shortest_wall:
                        shortest_wall = wall[-1]

                WallPlot.plot_mean(wall, value, label=used_label, linestyle=linestyle)
        if return_shortest_wall:
            return shortest_wall

    @staticmethod
    def linestyles():
        """Cycle through all different linestyles of `matplotlib`."""
        _linestyles = ["-", "--", "-.", ":"]
        for style in cycle(_linestyles):
            yield style

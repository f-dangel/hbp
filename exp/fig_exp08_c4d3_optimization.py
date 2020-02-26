"""Plot experiment of training c4d3 with sigmoid activations."""

from os import makedirs, path

import matplotlib
import matplotlib.pyplot as plt
from palettable.colorbrewer.sequential import PuRd_5, YlGnBu_5, YlOrRd_5

from exp.exp08_c4d3_optimization_adam import filenames as exp08_adam_files
from exp.exp08_c4d3_optimization_cvp import filenames as exp08_cvp_files
from exp.exp08_c4d3_optimization_kfac import filenames as exp08_kfac_files
from exp.exp08_c4d3_optimization_sgd import filenames as exp08_sgd_files
from exp.exp08_c4d3_optimization_sgd import parent_dir as exp08_dirname
from exp.plotting.plotting import OptimizationPlot, WallPlot
from exp.utils import directory_in_fig

matplotlib.use("agg")

# define colors
sgd_color = YlOrRd_5.mpl_colors[2]
adam_color = YlOrRd_5.mpl_colors[4]
cvp_colors = YlGnBu_5.mpl_colors[1:][::-1]
kfac_colors = PuRd_5.mpl_colors[2]  # Purples_5.mpl_colors[3]
colors = [sgd_color] + [adam_color] + [kfac_colors] + cvp_colors
# set color cycle
# matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colors)


def plot():
    # set color cycle
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=colors)

    # load data from exp08
    files_sgd = exp08_sgd_files()
    files_cvp = exp08_cvp_files()
    files_adam = exp08_adam_files()
    files_kfac = exp08_kfac_files()

    # set output directory
    fig_dir = directory_in_fig(exp08_dirname)
    # find metrics
    metrics = {m for (_, exp) in files_cvp.items() for m in exp.keys()}

    for metric in metrics:
        # output file
        this_fig_dir = fig_dir  # path.join(fig_dir, fig_sub)
        out_file = path.join(this_fig_dir, metric)
        makedirs(this_fig_dir, exist_ok=True)

        # create figure
        plt.figure()
        # plot sgd
        plot_labels = ["SGD"]
        plot_files = [files_sgd["SGD"][metric]]
        OptimizationPlot.create_standard_plot(
            "epoch",
            metric.replace("_", " "),
            plot_files,
            plot_labels,
            # scale by training set
            scale_steps=50000,
        )

        # plot Adam
        plot_labels = ["Adam"]
        plot_files = [files_adam["Adam"][metric]]
        OptimizationPlot.create_standard_plot(
            "epoch",
            metric.replace("_", " "),
            plot_files,
            plot_labels,
            # scale by training set
            scale_steps=50000,
        )

        # plot KFAC
        plot_labels = ["KFAC"]
        plot_files = [files_kfac["KFAC"][metric]]
        OptimizationPlot.create_standard_plot(
            "epoch",
            metric.replace("_", " "),
            plot_files,
            plot_labels,
            # scale by training set
            scale_steps=50000,
        )

        # collect plots for different curvature matrices
        plot_labels = [
            r"GGN, $\alpha_1$",
            r"GGN, $\alpha_2$",
            # 'PCH-abs1',
            "PCH-abs",
            "PCH-clip",
        ]
        # plot_labels = ['CG, GGN', 'CG, PCH (abs)', 'CG, PCH (clip)']
        plot_files = [files_cvp[l][metric] for l in plot_labels]
        idx = [2, 3, 0, 1]
        plot_files = [plot_files[i] for i in idx]
        plot_labels = [plot_labels[i] for i in idx]
        # plot CGN curvature matrices
        OptimizationPlot.create_standard_plot(
            "epoch",
            metric.replace("_", " "),
            plot_files,
            plot_labels,
            # scale by training set
            scale_steps=50000,
        )

        plt.legend()
        OptimizationPlot.save_as_tikz(out_file)
        OptimizationPlot.post_process(out_file)


def seed_filename(merged_filename, seed):
    seed_dir = "seed{}".format(seed)
    filename = path.basename(merged_filename)
    dirname = path.dirname(merged_filename)
    return path.join(dirname, seed_dir, filename)


def wall_plot(seeds):
    # set color cycle
    multiplied_colors = []
    # no SGD
    for c in colors[1:]:
        for _ in range(len(seeds)):
            multiplied_colors.append(c)
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=multiplied_colors)

    # load data from exp08
    files_cvp = exp08_cvp_files()
    files_adam = exp08_adam_files()
    files_kfac = exp08_kfac_files()

    # set output directory
    fig_dir = directory_in_fig(exp08_dirname)
    # find metrics
    metrics = {m for (_, exp) in files_adam.items() for m in exp.keys()}

    for metric in metrics:

        walls = []

        # output file
        this_fig_dir = fig_dir  # path.join(fig_dir, fig_sub)
        out_file = path.join(this_fig_dir, "wall_{}".format(metric))
        makedirs(this_fig_dir, exist_ok=True)

        # create figure
        plt.figure()

        # plot Adam
        plot_labels = ["Adam"]
        plot_files = [files_adam["Adam"][metric]]

        plot_files = [[seed_filename(f, s) for f in plot_files for s in seeds]]

        shortest_wall = WallPlot.create_standard_plot(
            "wall [s]", metric.replace("_", " "), plot_files, plot_labels
        )
        walls.append(shortest_wall)

        # plot KFAC
        plot_labels = ["KFAC"]
        plot_files = [files_kfac["KFAC"][metric]]

        plot_files = [[seed_filename(f, s) for f in plot_files for s in seeds]]

        # import pprint
        # print(pprint.pformat(plot_files))

        shortest_wall = WallPlot.create_standard_plot(
            "wall [s]", metric.replace("_", " "), plot_files, plot_labels
        )
        walls.append(shortest_wall)

        # # collect plots for different curvature matrices
        plot_labels = [
            r"GGN, $\alpha_1$",
            r"GGN, $\alpha_2$",
            # 'PCH-abs1',
            "PCH-abs",
            "PCH-clip",
        ]
        # plot_labels = ['CG, GGN', 'CG, PCH (abs)', 'CG, PCH (clip)']
        plot_files = [files_cvp[l][metric] for l in plot_labels]
        idx = [2, 3, 0, 1]
        # idx = [0, 1]
        plot_files = [plot_files[i] for i in idx]
        plot_labels = [plot_labels[i] for i in idx]

        plot_files_seeds = []
        for f in plot_files:
            plot_files_seeds.append([seed_filename(f, s) for s in seeds])

        # import pprint
        # print(pprint.pformat(plot_files_seeds))

        # plot CGN curvature matrices
        forget_plots = [0, 1, 3]
        shortest_wall = WallPlot.create_standard_plot(
            "wall [s]",
            metric.replace("_", " "),
            plot_files_seeds,
            plot_labels,
            forget=forget_plots,
        )
        walls.append(shortest_wall)

        plt.xlim(0, 400)
        plt.legend()
        WallPlot.save_as_tikz(out_file)
        WallPlot.post_process(out_file)


if __name__ == "__main__":
    plot()

    from exp.exp08_c4d3_optimization_adam import SEEDS as adam_seeds
    from exp.exp08_c4d3_optimization_cvp import SEEDS as cvp_seeds
    from exp.exp08_c4d3_optimization_kfac import SEEDS as kfac_seeds

    assert adam_seeds == cvp_seeds == kfac_seeds

    LINES_FOR_WALL = 10
    seeds = adam_seeds[:LINES_FOR_WALL]

    wall_plot(seeds)

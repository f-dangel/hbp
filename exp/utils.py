"""Utility functions for running experiments.

* directory to log quantities : ../dat
* directory to store figures : ../fig
"""

import math
from collections import OrderedDict
from os import path
from warnings import warn

from exp.training.runner import TrainingRunner

parent_dir = path.dirname(path.dirname(path.realpath(__file__)))


def run_training(labels, experiments, seeds=None):
    """Execute the experiments, return filenames of the merged runs."""
    if seeds is None:
        seeds = range(10)
    assert len(experiments) == len(labels)
    for train_fn in experiments:
        runner = TrainingRunner(train_fn)
        runner.run(seeds)


def merge_runs_return_files(labels, experiments, seeds=None):
    """Merge runs and return files of the merged data."""
    if seeds is None:
        seeds = range(10)
    filenames = OrderedDict()
    for label, train_fn in zip(labels, experiments):
        runner = TrainingRunner(train_fn)
        m_to_f = runner.merge_runs(seeds)
        filenames[label] = m_to_f
    return filenames


def directory_in_data(dir_name):
    """Return path in data folder with given name."""
    return path.join(parent_dir, "dat", dir_name)


def directory_in_fig(dir_name):
    """Return path in fig folder with given name."""
    return path.join(parent_dir, "fig", dir_name)


def dirname_from_params(**kwargs):
    """Concatenate key, value pairs alphabetically, split by underscore."""
    ordered = sorted(kwargs.items())
    words = ["_".join([key, str(value)]) for key, value in ordered]
    return "_".join(words)


def run_directory_exists(logdir):
    """Return warning: Run directory exists, will be skipped."""
    if path.isdir(logdir):
        warn(
            "\nLogging directory already exists:\n{}\n"
            "It is likely that this run will be skipped.\n".format(logdir)
        )
        return True
    return False


def boxed_message(message):
    """Draw a box around a message.

    Parameters:
    -----------
    message : str
        Message to be boxed

    Returns:
    --------
    str
        Boxed message

    References:
    -----------
    - https://github.com/quapka/expecto-anything/blob/master/boxed_msg.py
    """

    def format_line(line, max_length):
        half_diff = (max_length - len(line)) / 2
        return "{}{}{}{}{}".format(
            "| ", " " * math.ceil(half_diff), line, " " * math.floor(half_diff), " |\n"
        )

    lines = message.split("\n")
    max_length = max(len(l) for l in lines)
    horizontal = "{}{}{}".format("+", "-" * (max_length + 2), "+\n")
    result = horizontal
    for l in lines:
        result += format_line(l, max_length)
    result += horizontal
    return result.strip()

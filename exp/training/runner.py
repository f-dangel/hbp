"""Run experiments with different seeds."""

from os import path
from warnings import warn

from bpexts.utils import set_seeds

from .merger import CSVMerger


class TrainingRunner:
    """Run experiments with different seeds."""

    def __init__(self, training_fn):
        """
        Parameters:
        -----------
        training_fn : (function)
            Function that creates an instance of `Training` (see
            `training.py`
        """
        self.training_fn = training_fn

    def run(self, seeds):
        """Run training procedure with different seeds, extract to CSV.

        Parameters:
        -----------
        seeds : (list(int))
            List with init values for random seed
        num_epochs : (int)
            Number of epochs to train on
        device : (torch.Device)
           Device to run the training on
        logs_per_epoch : (int)
            Number of datapoints written into logdir per epoch
        """
        for seed in seeds:
            set_seeds(seed)
            training = self.training_fn()
            sub = self._seed_dir_name(seed)
            if path.isdir(training.logger_subdir_path(sub)):
                warn(
                    "Logging directory {} already exists. Skipping"
                    " run".format(training.logger_subdir_path(sub))
                )
            else:
                training.point_logger_to_subdir(sub)
                training.run()
                training.logger.extract_logdir_to_csv()

    def merge_runs(self, seeds, metrics=None):
        """Merge metrics from runs to .csv and compute mean/std.

        Creates file `metric.csv` in the train function's log directory.

        Parameters:
        ----------
        seeds : (list(int))
            Initial seeds of the runs to be merged 
        metrics : (list(string))
            Metrics to average over:
            * `'batch_acc'`: Batch accuracy
            * `'batch_loss'`: Batch loss 
            * `'test_acc'`: Accuracy on test set
            * `'test_loss'`: Loss on test set
            * `'train_acc'`: Accuracy on a subset of the train set
            * `'train_loss'`: Loss on a subset of the train set
            Use all available metrics if left `None`

        Returns:
        --------
        (dict)
            Dictionary with metric names as keys and merged output
            file as value
        """
        metrics = (
            [
                "batch_acc",
                "batch_loss",
                "test_acc",
                "test_loss",
                "train_acc",
                "train_loss",
            ]
            if metrics is None
            else metrics
        )
        metric_to_file = {}
        for metric in metrics:
            df = self._merge_metric(metric, seeds)
            out_file = path.join(self._logdir_name(), "{}.csv".format(metric))
            df.to_csv(out_file, index=False)
            metric_to_file[metric] = out_file
        return metric_to_file

    def _merge_metric(self, metric, seeds):
        """Merge metric of different runs, add mean and std, return df."""
        logdir = self._logdir_name()
        source_files = [
            path.join(self._seed_dir_name(seed, logdir), "{}.csv".format(metric))
            for seed in seeds
        ]
        labels = [self._seed_dir_name(seed) for seed in seeds]
        return CSVMerger(source_files, labels).merge()

    def _logdir_name(self):
        """Return the log directory of the training instance."""
        return self.training_fn().logdir

    def _seed_dir_name(self, seed, parent_path=None):
        """Return name of subdirectory for a certain seed.

        Parameters:
        -----------
        seed : (int)
            Random seed of the run
        parent_path : (str)
            Path the seed directory is appended to
        """
        parent_path = "" if parent_path is None else parent_path
        return path.join(parent_path, "seed{}".format(seed))

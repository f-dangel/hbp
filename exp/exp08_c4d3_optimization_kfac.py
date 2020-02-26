"""Training of c4d3 on CIFAR-10 with Adam."""

from os import path

import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh

from exp.loading.load_cifar10 import CIFAR10Loader
from exp.models.convolution import cifar10_c4d3
from exp.third_party.optimizers.kfac import KFACOptimizer
from exp.training.second_order import KFACTraining
from exp.utils import (
    directory_in_data,
    dirname_from_params,
    merge_runs_return_files,
    run_training,
)

# directories
dirname = "exp08_c4d3_optimization/kfac"
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logs_per_epoch = 4
test_batch = 100

# mapping from strings to activation functions
activation_dict = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


def cifar10_kfac_train_fn(
    batch,
    activation,
    lr,
    momentum,
    stat_decay,
    damping,
    kl_clip,
    weight_decay,
    TCov,
    TInv,
    batch_averaged,
):
    """Create training instance for CIFAR10 KFAC experiment."""
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation,
        opt="kfac",
        batch=batch,
        lr=lr,
        mom=momentum,
        stat_dec=stat_decay,
        damp=damping,
        weight_dec=weight_decay,
        kl_clip=kl_clip,
        TCov=TCov,
        TInv=TInv,
        batch_avg=batch_averaged,
    )
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # set up training and run
        act = activation_dict[activation]
        model = cifar10_c4d3(conv_activation=act, dense_activation=act)
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch, test_batch_size=test_batch)
        optimizer = KFACOptimizer(
            model=model,
            lr=lr,
            momentum=momentum,
            stat_decay=stat_decay,
            damping=damping,
            kl_clip=kl_clip,
            weight_decay=weight_decay,
            TCov=TCov,
            TInv=TInv,
            batch_averaged=batch_averaged,
        )
        # initialize training (no second backward pass needed)
        train = KFACTraining(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            epochs,
            logs_per_epoch=logs_per_epoch,
            device=device,
        )
        return train

    return training_fn


def kfac_grid_search():
    """Define the grid search over the hyperparameters of KFAC."""
    # grid search: [200, 500, 1000]
    batch_sizes = [500]
    # grid search: [0.0001, 0.001, 0.01, 0.1, 1, 10]
    lrs = [0.1]

    momentum = 0.9
    activation = "sigmoid"
    stat_decay = 0.95
    damping = 0.001
    kl_clip = 0.001
    weight_decay = 0
    TCov = 10
    TInv = 100
    batch_averaged = True

    return [
        cifar10_kfac_train_fn(
            batch,
            activation,
            lr,
            momentum,
            stat_decay,
            damping,
            kl_clip,
            weight_decay,
            TCov,
            TInv,
            batch_averaged,
        )
        for batch, lr in zip(batch_sizes, lrs)
    ]


SEEDS = list(range(10))


def main():
    labels = [
        "KFAC",
    ]
    experiments = kfac_grid_search()

    run_training(labels, experiments, SEEDS)
    return merge_runs_return_files(labels, experiments, SEEDS)


def filenames():
    return main()


if __name__ == "__main__":
    main()

"""Experiments performed in Chen et al.: BDA-PCH, figure 2.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

from os import path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from bpexts.hbp.crossentropy import HBPCrossEntropyLoss
from bpexts.optim.cg_newton import CGNewton
from exp.loading.load_cifar10 import CIFAR10Loader
from exp.models.chen2018 import cifar10_model, hbp_cifar10_model
from exp.training.first_order import FirstOrderTraining
from exp.training.second_order import HBPSecondOrderTraining
from exp.utils import (
    directory_in_data,
    dirname_from_params,
    merge_runs_return_files,
    run_training,
)

# global hyperparameters
batch = 500
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dirname = "exp01_reproduce_chen_figures/cifar10"
data_dir = directory_in_data(dirname)
logs_per_epoch = 1


def cifar10_sgd_train_fn():
    """Create training instance for CIFAR10 SGD experiment."""
    # hyper parameters
    # ----------------
    lr = 0.1
    momentum = 0.9

    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(opt="sgd", batch=batch, lr=lr, mom=momentum)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # setting up training and run
        model = cifar10_model()
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch, test_batch_size=batch)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        # initialize training
        train = FirstOrderTraining(
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


def cifar10_cgnewton_train_fn(modify_2nd_order_terms):
    """Create training instance for CIFAR10 CG experiment

    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    """
    # hyper parameters
    # ----------------
    lr = 0.1
    alpha = 0.02
    cg_maxiter = 50
    cg_tol = 0.1
    cg_atol = 0

    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        opt="cgn",
        batch=batch,
        lr=lr,
        alpha=alpha,
        maxiter=cg_maxiter,
        tol=cg_tol,
        atol=cg_atol,
        mod2nd=modify_2nd_order_terms,
    )
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # set up training and run
        model = hbp_cifar10_model(
            average_input_jacobian=True, average_parameter_jacobian=True
        )
        loss_function = HBPCrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch, test_batch_size=batch)
        optimizer = CGNewton(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            cg_atol=cg_atol,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
        # initialize training
        train = HBPSecondOrderTraining(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            epochs,
            modify_2nd_order_terms,
            logs_per_epoch=logs_per_epoch,
            device=device,
        )
        return train

    return training_fn


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(10)
    labels = [
        "SGD",
        "CG (GGN)",
        "CG (PCH, abs)",
        "CG (PCH, clip)",
    ]
    experiments = [
        # 1) SGD curve
        cifar10_sgd_train_fn(),
        # 2) Generalized Gauss-Newton curve
        cifar10_cgnewton_train_fn("zero"),
        # 3) BDA-PCH curve
        cifar10_cgnewton_train_fn("abs"),
        # 4) alternative BDA-PCH curve
        cifar10_cgnewton_train_fn("clip"),
    ]
    run_training(labels, experiments, seeds)
    return merge_runs_return_files(labels, experiments, seeds)


def filenames():
    return main()


if __name__ == "__main__":
    main()

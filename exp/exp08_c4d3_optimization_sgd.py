"""Training of c4d3 on CIFAR-10 with SGD."""

from os import path

import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from torch.optim import SGD

from exp.loading.load_cifar10 import CIFAR10Loader
from exp.models.convolution import cifar10_c4d3
from exp.training.first_order import FirstOrderTraining
from exp.utils import (
    directory_in_data,
    dirname_from_params,
    merge_runs_return_files,
    run_training,
)

# directories
parent_dir = "exp08_c4d3_optimization/"
dirname = path.join(parent_dir, "sgd")
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logs_per_epoch = 4
test_batch = 100

# mapping from strings to activation functions
activation_dict = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


def cifar10_sgd_train_fn(batch, lr, momentum, activation):
    """Create training instance for CIFAR-10 SGD optimization.

    Parameters:
    -----------
    lr : float
        Learning rate for SGD
    momentum : float
        Momentum for SGD
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation, opt="sgd", batch=batch, lr=lr, mom=momentum
    )
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        act = activation_dict[activation]
        model = cifar10_c4d3(conv_activation=act, dense_activation=act)
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch, test_batch_size=test_batch)
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


def sgd_grid_search():
    r"""Define the grid search over the hyperparameters of SGD.

    Note on 'best' parameter:
    -------------------------
    SGD is not possible to optimize this net, so choose the parameter
    set which is close to Adam
    """
    # grid search: ['sigmoid']
    activations = ["sigmoid"]
    # grid search: [100, 200, 500]
    batch_sizes = [100]
    # grid search: numpy.logspace(-3, 1, 5)
    lrs = [0.001]
    # grid search: [0, 0.45, 0.9]
    momenta = [0.9]
    return [
        cifar10_sgd_train_fn(
            batch=batch, lr=lr, momentum=momentum, activation=activation
        )
        for batch in batch_sizes
        for lr in lrs
        for momentum in momenta
        for activation in activations
    ]


SEEDS = list(range(10))


def main():
    """Execute the experiments, return filenames of the merged runs."""
    labels = ["SGD"]
    experiments = sgd_grid_search()

    run_training(labels, experiments, SEEDS)
    return merge_runs_return_files(labels, experiments, SEEDS)


def filenames():
    return main()


if __name__ == "__main__":
    main()

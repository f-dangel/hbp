"""Training of c4d3 on CIFAR-10 with Adam."""

from os import path

import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from torch.optim import Adam

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
dirname = "exp08_c4d3_optimization/adam"
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logs_per_epoch = 4
test_batch = 100

# mapping from strings to activation functions
activation_dict = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


def cifar10_adam_train_fn(batch, lr, betas, activation):
    """Create training instance for CIFAR-10 Adam optimization.

    Parameters:
    -----------
    lr : float
        Learning rate for Adam
    betas : (float, float)
        Coefficients for computing running averages in Adam
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation, opt="adam", batch=batch, lr=lr, b1=betas[0], b2=betas[1]
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
        optimizer = Adam(model.parameters(), lr=lr, betas=betas)
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


def adam_grid_search():
    """Define the grid search over the hyperparameters of Adam."""
    # grid search:  ['sigmoid']
    activations = ["sigmoid"]
    # grid search: [100, 200, 500]
    batch_sizes = [100]
    # grid search: numpy.logspace(-4, 1, 6)
    lrs = [0.001]
    # grid search:  [(0.9, 0.999)]
    betas = [(0.9, 0.999)]
    return [
        cifar10_adam_train_fn(
            batch=batch, lr=lr, betas=beta_pair, activation=activation
        )
        for batch in batch_sizes
        for lr in lrs
        for beta_pair in betas
        for activation in activations
    ]


SEEDS = list(range(10))


def main():
    labels = ["Adam"]
    experiments = adam_grid_search()

    run_training(labels, experiments, SEEDS)
    return merge_runs_return_files(labels, experiments, SEEDS)


def filenames():
    return main()


if __name__ == "__main__":
    main()

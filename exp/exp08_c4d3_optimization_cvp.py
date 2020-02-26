"""Training of c4d3 on CIFAR-10 with CGN and CVP."""

from os import path

import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh

from bpexts.cvp.sequential import convert_torch_to_cvp
from bpexts.optim.cg_newton import CGNewton
from exp.loading.load_cifar10 import CIFAR10Loader
from exp.models.convolution import cifar10_c4d3
from exp.training.second_order import CVPSecondOrderTraining
from exp.utils import (
    directory_in_data,
    dirname_from_params,
    merge_runs_return_files,
    run_training,
)

# directories
dirname = "exp08_c4d3_optimization/cvp"
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logs_per_epoch = 4
test_batch = 100

# mapping from strings to activation functions
activation_dict = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


def cifar10_cgnewton_train_fn(
    batch, modify_2nd_order_terms, activation, lr, alpha, cg_maxiter, cg_tol, cg_atol
):
    """Create training instance for CIFAR10 CG experiment.

    Parameters:
    -----------
    batch : int
        Batch size
    modify_2nd_order_terms : str
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    lr : float
        Learning rate
    alpha : float, between 0 and 1
        Regularization in HVP, see Chen paper for more details
    cg_maxiter : int
        Maximum number of iterations for CG
    cg_tol : float
        Relative tolerance for convergence of CG
    cg_atol : float
        Absolute tolerance for convergence of CG
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation,
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
        act = activation_dict[activation]
        model = cifar10_c4d3(conv_activation=act, dense_activation=act)
        model = convert_torch_to_cvp(model)
        loss_function = convert_torch_to_cvp(CrossEntropyLoss())
        data_loader = CIFAR10Loader(train_batch_size=batch, test_batch_size=test_batch)
        optimizer = CGNewton(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            cg_atol=cg_atol,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
        # initialize training
        train = CVPSecondOrderTraining(
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


def cgn_grid_search():
    """Define the grid search over the hyperparameters of SGD."""
    # grid search: [200, 500, 1000]
    batch_sizes = [1000]
    # grid search: ['zero', 'abs', 'clip']
    mod2nds = ["zero", "zero", "abs", "clip"]
    # grid search: ['sigmoid']
    activations = ["sigmoid"]
    # grid search: [0.05, 0.1, 0.2,]
    lrs = [0.1, 0.1, 0.2, 0.1]
    # grid search: [0.0001, 0.001, 0.01, 0.1]
    alphas = [0.0001, 0.001, 0.001, 0.0001]
    cg_atol = 0.0
    cg_maxiter = 200
    cg_tols = [1e-1]
    return [
        cifar10_cgnewton_train_fn(
            batch=batch,
            modify_2nd_order_terms=mod2nd,
            activation=activation,
            lr=lr,
            alpha=alpha,
            cg_maxiter=cg_maxiter,
            cg_tol=cg_tol,
            cg_atol=cg_atol,
        )
        for cg_tol in cg_tols
        for activation in activations
        for batch in batch_sizes
        for mod2nd, lr, alpha, in zip(mod2nds, lrs, alphas)
    ]


SEEDS = list(range(10))


def main():
    labels = [
        r"GGN, $\alpha_1$",
        r"GGN, $\alpha_2$",
        "PCH-abs",
        "PCH-clip",
    ]
    experiments = cgn_grid_search()

    run_training(labels, experiments, SEEDS)
    return merge_runs_return_files(labels, experiments, SEEDS)


def filenames():
    return main()


if __name__ == "__main__":
    main()

"""CNNs for testing/experiments."""


import torch
import torch.nn as nn
from backpack.core.layers import Flatten

from deepobs.pytorch.testproblems import testproblems_modules


def cifar10_c4d3(conv_activation=nn.ReLU, dense_activation=nn.ReLU):
    """CNN for CIFAR-10 dataset with 4 convolutional and 3 fc layers.


    Modified from:
    https://github.com/Zhenye-Na/deep-learning-uiuc/tree/master/assignments/mp3
    (remove Dropout, Dropout2d and BatchNorm2d)
    """
    return nn.Sequential(
        # Conv Layer block 1
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        conv_activation(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Conv Layer block 2
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        conv_activation(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Flatten
        Flatten(),
        # Dense layers
        nn.Linear(2048, 512),
        dense_activation(),
        nn.Linear(512, 64),
        dense_activation(),
        nn.Linear(64, 10),
    )


def deepobs_cifar10_c3d3(conv_activation=nn.ReLU, dense_activation=nn.ReLU):
    """3c3d network from DeepOBS.

    The weight matrices are initialized using Xavier initialization and the
    biases are initialized to zero.
    """

    def all_children(sequential):
        children = []
        for child in sequential.children():
            if isinstance(child, nn.Sequential):
                children += all_children(child)
            else:
                children.append(child)
        return children

    def replace_activations(c3d3):
        "Replace ReLUs with specified activations for conv and dense."
        # conv activations
        c3d3.relu1 = conv_activation()
        c3d3.relu2 = conv_activation()
        c3d3.relu3 = conv_activation()
        # dense activations
        c3d3.relu4 = dense_activation()
        c3d3.relu5 = dense_activation()

        return c3d3

    def replace_deepobs_flatten(c3d3):
        """Replace DeepOBS flatten with bpexts Flatten."""
        c3d3.flatten = Flatten()
        return c3d3

    def set_tf_same_hyperparams(c3d3):
        """Forward pass to set the hyperparams of padding and max pooling
        in tensorflow 'same' mode."""
        CIFAR10_TEST_SHAPE = (1, 3, 32, 32)
        input = torch.rand(CIFAR10_TEST_SHAPE)
        _ = c3d3(input)
        return c3d3

    num_outputs = 10

    c3d3 = testproblems_modules.net_cifar10_3c3d(num_outputs)
    c3d3 = set_tf_same_hyperparams(c3d3)
    c3d3 = replace_activations(c3d3)
    c3d3 = replace_deepobs_flatten(c3d3)

    modules = all_children(c3d3)

    return nn.Sequential(*modules)

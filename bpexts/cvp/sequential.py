"""
Curvature-vector products for sequences of modules and conversion of torch.nn
 layers to CVP layers.
"""

from backpack.core.layers import Flatten
# torch layers
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from ..hbp.module import hbp_decorate
from .conv2d import CVPConv2d
from .crossentropy import CVPCrossEntropyLoss
from .flatten import CVPFlatten
from .linear import CVPLinear
from .maxpool2d import CVPMaxPool2d
from .padding import CVPZeroPad2d
# CVP layers
from .relu import CVPReLU
from .sigmoid import CVPSigmoid
from .tanh import CVPTanh


class CVPSequential(hbp_decorate(Sequential)):
    """Sequence of modules with recursive Hessian-vector products."""

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Sequential):
            raise ValueError(
                "Expecting torch.nn.Sequential, got {}".format(torch_layer.__class__)
            )
        layers = []
        for mod in torch_layer:
            layers.append(convert_torch_to_cvp(mod))
        return cls(*layers)

    # override
    def enable_hbp(self):
        for mod in self.children():
            mod.enable_hbp()

    # override
    def hbp_hooks(self):
        """No hooks required."""

    # override
    def backward_hessian(
        self, output_hessian, compute_input_hessian=True, modify_2nd_order_terms="none"
    ):
        """Propagate Hessian-vector product through the network.

        Starting from the last layer, call `backward_hessian` recursively
        until ending up in the first module.
        """
        out_h = output_hessian
        for idx in reversed(range(len(self))):
            module = self[idx]
            compute_in = True if (idx != 0) else compute_input_hessian
            out_h = module.backward_hessian(
                out_h,
                compute_input_hessian=compute_in,
                modify_2nd_order_terms=modify_2nd_order_terms,
            )
        return out_h


# supported conversions
conversions = [
    (ReLU, CVPReLU),
    (Sigmoid, CVPSigmoid),
    (Tanh, CVPTanh),
    (Linear, CVPLinear),
    (Conv2d, CVPConv2d),
    (MaxPool2d, CVPMaxPool2d),
    (ZeroPad2d, CVPZeroPad2d),
    (Sequential, CVPSequential),
    (Flatten, CVPFlatten),
    (CrossEntropyLoss, CVPCrossEntropyLoss),
]


def convert_torch_to_cvp(layer):
    """Convert torch layer to corresponding CVP layer."""
    for (torch_cls, cvp_cls) in conversions:
        if isinstance(layer, torch_cls):
            return cvp_cls.from_torch(layer)
    _print_conversions()
    raise ValueError("Class {} cannot be converted to CVP.".format(layer.__class__))


def _print_conversions():
    """Print all possible conversions."""
    print("Supported conversions:")
    for torch_cls, cvp_cls in conversions:
        print("{}\t->\t{}".format(torch_cls.__name__, cvp_cls.__name__))

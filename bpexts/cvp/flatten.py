"""Curvature-vector products of batch-wise flatten operation."""

from backpack.core.layers import Flatten

from ..hbp.module import hbp_decorate


class CVPFlatten(hbp_decorate(Flatten)):
    """Flatten all dimensions except batch dimension, with CVP supoprt."""

    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Flatten):
            raise ValueError(
                "Expecting backpack.core.layers.Flatten, got {}".format(
                    torch_layer.__class__
                )
            )
        return cls()

    # override
    def hbp_hooks(self):
        """No hooks required."""

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms="none"):
        """Pass on the Hessian with respect to the layer input."""
        return output_hessian

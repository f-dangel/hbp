"""Curvature-vector products for linear layer."""

from torch import diag, einsum
from torch.nn import CrossEntropyLoss, functional

from ..hbp.module import hbp_decorate


class HBPCrossEntropyLoss(hbp_decorate(CrossEntropyLoss)):
    """Cross-entropy loss with recursive Hessian-vector products."""

    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        if weight is not None:
            raise NotImplementedError("Only supports weight = None")
        if ignore_index != -100:
            raise NotImplementedError("Only supports ignore_index = -100")
        if reduction != "mean":
            raise NotImplementedError(r"Only supports reduction = 'mean'")
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, CrossEntropyLoss):
            raise ValueError(
                "Expecting torch.nn.CrossEntropyLoss, got {}".format(
                    torch_layer.__class__
                )
            )
        # create instance
        cross_entropy = cls(
            weight=torch_layer.weight,
            ignore_index=torch_layer.ignore_index,
            reduction=torch_layer.reduction,
        )
        return cross_entropy

    # override
    def hbp_hooks(self):
        """Install hooks to track quantities required for CVP."""
        self.register_exts_forward_pre_hook(self.compute_and_store_softmax)

    # --- hooks ---
    @staticmethod
    def compute_and_store_softmax(module, input):
        """Compute and save softmax of layer input.

        Intended use as pre-forward hook.
        Initialize module buffer 'input_softmax'.
        """
        if not len(input) == 2:
            raise ValueError("Wrong number of inputs")
        assert len(tuple(input[0].size())) == 2
        input_softmax = functional.softmax(input[0].detach(), dim=1)
        assert input_softmax.size() == input[0].size()
        module.register_exts_buffer("input_softmax", input_softmax)

    # --- end of hooks ---

    def batch_summed_hessian(self, loss, output):
        """Return batch-summed Hessian with respect to the input."""
        probs = self.input_softmax
        batch, num_classes = probs.size()

        sum_H = diag(probs.sum(0)) - einsum("bi,bj->ij", (probs, probs))
        return sum_H / batch

"""Test HBP decoration of torch.nn.Module subclasses."""

from torch.nn import Linear

from bpexts.hbp.module import hbp_decorate


def test_hbp_decorate():
    """Test decoration of a linear layer for HBP."""
    modifiedLinear = hbp_decorate(Linear)
    inputs, outputs = 5, 2
    hbp_linear = modifiedLinear(in_features=inputs, out_features=outputs)
    assert (hbp_linear.__class__.__name__) == "HBPLinear"

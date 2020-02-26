"""Test batch-averaged Hessian computation."""

from torch import allclose, tensor

from bpexts.hbp.loss import batch_summed_hessian


def example_loss_function(x):
    """Sum up all square elements of a tensor."""
    return (x ** 2).view(-1).sum(0)


def test_batch_averaged_hessian():
    """Test batch-averaged Hessian."""
    # 2 samples of 3d vectors
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    f = example_loss_function(x)
    result = tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    avg_hessian = batch_summed_hessian(f, x)
    assert allclose(result, avg_hessian)

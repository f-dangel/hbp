def torch_contains_nan(tensor):
    """Return whether a tensor contains NaNs.

    Parameters
    ----------
    tensor : :obj:`torch.Tensor`
        Tensor to be checked for NaNs.

    Returns
    -------
    bool
        If at least one NaN is contained in :obj:`tensor`.
    """
    return any(tensor.view(-1) != tensor.view(-1))

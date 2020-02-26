"""Utility functions."""

import gc
import random

import numpy
import torch
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.utils import vector_to_parameter_list


def flatten_and_concatenate(list_of_tensors):
    flat = [t.contiguous().view(-1) for t in list_of_tensors]
    return torch.cat(flat)


def exact_hessian(f, parameters):
    r"""Compute all second derivatives of a scalar w.r.t. `parameters`.

    The order of parameters corresponds to a one-dimensional
    vectorization followed by a concatenation of all tensors in
    `parameters`.

    Parameters
    ----------
    f : scalar torch.Tensor
        Scalar PyTorch function/tensor.
    parameters : list or tuple or iterator of torch.Tensor
        Iterable object containing all tensors acting as variables of `f`.

    Returns
    -------
    torch.Tensor
        Hessian of `f` with respect to the concatenated version
        of all flattened quantities in `parameters`

    Note
    ----
    The parameters in the list are all flattened and concatenated
    into one large vector `theta`. Return the matrix :math:`d^2 E /
    d \theta^2` with

    .. math::

        (d^2E / d \theta^2)[i, j] =  (d^2E / d \theta[i] d \theta[j]).
    """
    params = list(parameters)
    dim = sum(p.numel() for p in params)

    def hvp(v):
        vecs = vector_to_parameter_list(v, params)
        results = hessian_vector_product(f, params, vecs)
        return flatten_and_concatenate(results)

    return matrix_from_mvp(hvp, (dim, dim), device=f.device)


def exact_hessian_diagonal_blocks(f, parameters):
    """Compute diagonal blocks of a scalar function's Hessian.

    Parameters
    ----------
    f : scalar of torch.Tensor
        Scalar PyTorch function
    parameters : list or tuple or iterator of torch.Tensor
        List of parameters whose second derivatives are to be computed
        in a blockwise manner

    Returns
    -------
    list of torch.Tensor
        Hessian blocks. The order is identical to the order specified
        by `parameters`

    Note
    ----
    For each parameter, `exact_hessian` is called.
    """
    return [exact_hessian(f, [p]) for p in parameters]


def matrix_from_mvp(mvp, dims, device=None):
    """Compute matrix representation from matrix-vector product.

    Parameters:
    -----------
    mvp : function
        Matrix-vector multiplication routine.
    dims : tuple(int)
        Dimensions of the matrix

    Returns:
    --------
    torch.Tensor
        Matrix representation on ``device``
    """
    if device is None:
        device = torch.device("cpu")

    assert len(dims) == 2
    matrix = torch.zeros(*dims).to(device)
    for i in range(dims[1]):
        v = torch.zeros(dims[0]).to(device)
        v[i] = 1.0
        matrix[:, i] = mvp(v)
    return matrix


def set_seeds(seed=None):
    """Set random seeds of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy`,
    :mod:`random`.

    Per default, no reset will be performed.

    Parameters
    ----------
    seed : :obj:`int` or :obj:`None`, optional
        Seed initialization value, no reset if unspecified
    """
    if seed is not None:
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # NumPy
        numpy.random.seed(seed)
        # random
        random.seed(seed)


def memory_report():
    """Report the memory usage of the :obj:`torch.tensor.storage` both
    on CPUs and GPUs.

    Returns
    -------
    tuple
        Two tuples, each consisting of the number of allocated tensor
        elements and the total storage in MB on GPU and CPU, respectively.

    Notes
    -----
    * The code is a modified version from the snippet provided by
      https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
    """

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type. Ignore sparse tensors.

        There are two major storage types in major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory

        Parameters
        ----------
        tensors : (list(torch.Tensor))
            The tensors of specified type
        mem_type : (str)
            'CPU' or 'GPU' in current implementation

        Returns
        -------
        total_numel, total_mem : (int, float)
            Total number of allocated elements and total memory reserved
        """
        print("Storage on {}\n{}".format(mem_type, "-" * LEN))
        total_numel, total_mem, visited_data = 0, 0.0, []

        # sort by size
        sorted_tensors = sorted(tensors, key=lambda t: t.storage().data_ptr())

        for tensor in sorted_tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()

            numel = tensor.storage().size()
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024.0 ** 2  # 32bit = 4Byte, MByte
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print("{}  \t{}\t\t{:.7f}\t\t{}".format(element_type, size, mem, data_ptr))

            if data_ptr not in visited_data:
                total_numel += numel
                total_mem += mem
            visited_data.append(data_ptr)

        print(
            "{}\nTotal Tensors (not counting shared multiple times):"
            "{}\nUsed Memory Space: {:.7f} MB\n{}".format(
                "-" * LEN, total_numel, total_mem, "-" * LEN
            )
        )
        return total_numel, total_mem

    gc.collect()
    LEN = 65
    print("=" * LEN)
    objects = gc.get_objects()
    print("{}\t{}\t\t\t{}".format("Element type", "Size", "Used MEM(MB)"))

    tensors = []
    for obj in objects:
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensors.append(obj)
        except Exception:
            pass

    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    gpu_stats = _mem_report(cuda_tensors, "GPU")
    cpu_stats = _mem_report(host_tensors, "CPU")
    print("=" * LEN)

    return gpu_stats, cpu_stats

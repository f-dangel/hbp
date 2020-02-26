"""Hessian backpropagation for 2D convolution."""

from numpy import prod
from torch import arange, einsum, tensor, zeros
from torch.nn import Conv2d, functional

from .module import hbp_decorate


class HBPConv2d(hbp_decorate(Conv2d)):
    """2D Convolution with Hessian backpropagation functionality."""

    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Conv2d):
            raise ValueError(
                "Expecting torch.nn.Conv2d, got {}".format(torch_layer.__class__)
            )
        # create instance
        conv2d = cls(
            torch_layer.in_channels,
            torch_layer.out_channels,
            torch_layer.kernel_size,
            stride=torch_layer.stride,
            padding=torch_layer.padding,
            dilation=torch_layer.dilation,
            groups=torch_layer.groups,
            bias=torch_layer.bias is not None,
        )
        # copy parameters
        conv2d.weight = torch_layer.weight
        conv2d.bias = torch_layer.bias
        return conv2d

    # override
    def set_hbp_approximation(
        self, average_input_jacobian=None, average_parameter_jacobian=True
    ):
        """Not sure if useful to implement"""
        if average_parameter_jacobian is False:
            raise NotImplementedError
        super().set_hbp_approximation(
            average_input_jacobian=None,
            average_parameter_jacobian=average_parameter_jacobian,
        )

    # override
    def hbp_hooks(self):
        """Install hook storing unfolded input."""
        self.register_exts_forward_pre_hook(
            self.store_mean_unfolded_input_and_sample_dimension
        )

    def unfold(self, input):
        """Unfold input using convolution hyperparameters."""
        return functional.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    # --- hooks ---
    @staticmethod
    def store_mean_unfolded_input_and_sample_dimension(module, input):
        """Save mean of unfolded input and dimension of input sample.

        Indended use as pre-forward hook.
        Initialize module buffer 'mean_unfolded_input'.
        Initialize module buffer 'sample_dim'.
        """
        if not len(input) == 1:
            raise ValueError("Cannot handle multi-input scenario")
        mean_input = input[0].mean(0).unsqueeze(0).detach()
        mean_unfolded_input = module.unfold(mean_input)[0, :]
        module.register_exts_buffer("mean_unfolded_input", mean_unfolded_input)
        # save number of elements in a single sample
        sample_dim = tensor(input[0].size()[1:])
        module.register_exts_buffer("sample_dim", sample_dim)

    # --- end of hooks ---

    # override
    def parameter_hessian(self, output_hessian):
        """Compute parameter Hessian.

        The Hessian of the bias (if existent) is stored in the attribute
        self.bias.hessian. Hessian-vector product function is stored in
        self.bias.hvp.

        The Hessian of the weight is not computed explicitely for memory
        efficiency. Instead, a method is stored in self.weight.hessian,
        that produces the explicit Hessian matrix when called. Hessian-
        vector product function is stored in self.weight.hvp.
        """
        if self.bias is not None:
            self.init_bias_hessian(output_hessian.detach())
        self.init_weight_hessian(output_hessian.detach())

    # override
    def input_hessian(
        self, output_hessian, compute_input_hessian=True, modify_2nd_order_terms="none"
    ):
        """Compute the Hessian with respect to the layer input."""
        if compute_input_hessian is False:
            return None
        else:
            unfolded_hessian = self.unfolded_input_hessian(output_hessian.detach())
            return self.sum_shared_inputs(unfolded_hessian)

    def sum_shared_inputs(self, unfolded_input_hessian):
        """Sum rows and columns belonging to the same original input.

        The unfolding procedure of the input during the forward pass
        of the convolution corresponds to spreading out the input
        into a larger matrix. Given the Hessian of the loss with
        respect to the unfolded input, the Hessian of the original
        input is obtained by summing up rows and columns of the positions
        where certain input has been spread.
        """
        sample_numel = int(prod(self.sample_dim.numpy()))
        idx_num = sample_numel + 1
        # index map from input to unfolded input
        idx_unfolded = self._unfolded_index_map().view(-1)
        # sum rows of all positions an input was unfolded to
        acc_rows = zeros(
            idx_num,
            unfolded_input_hessian.size()[1],
            device=unfolded_input_hessian.device,
            dtype=unfolded_input_hessian.dtype,
        )
        acc_rows.index_add_(0, idx_unfolded, unfolded_input_hessian)
        # sum columns of all positions an input was unfolded to
        acc_cols = zeros(
            idx_num,
            idx_num,
            device=unfolded_input_hessian.device,
            dtype=unfolded_input_hessian.dtype,
        )
        acc_cols.index_add_(1, idx_unfolded, acc_rows)
        # cut out dimension of padding elements (index 0)
        return acc_cols[1:, 1:]

    def _unfolded_index_map(self):
        """Return the index map from input image to unfolded image.

        Padded elements are assigned the index 0, and the pixels
        from the input image are enumerated in ascending order starting
        from 1.

        Returns:
        --------
        torch.LongTensor
            Tensor of the same size as an unfolded image containing
            the indices of the original image.
        """
        sample_numel = int(prod(self.sample_dim.numpy()))
        idx_num = sample_numel + 1
        # input image with pixels containing the index value (starting from 1)
        # also take padding into account (will be indicated by index 0)
        # NOTE: Only works for padding with zeros!!
        idx = arange(1, idx_num, device=self.mean_unfolded_input.device).view(
            (1,) + tuple(self.sample_dim),
        )
        # unfolded indices (indicate which input unfolds to which index)
        idx_unfolded = self.unfold(idx.float()).long()
        return idx_unfolded

    def unfolded_input_hessian(self, out_h):
        """Compute Hessian with respect to the layer's unfolded input.

        Make use of the relation between the output and the unfolded
        input by a matrix multiplication with the kernel matrix. Hence
        the Jacobian is given by a Kronecker product, which has to be
        multiplied to the output Hessian from left and right. Unfortunately,
        this cannot be simplified further, resultin in a tensor network
        that has to be contracted (see the `einsum` call below).
        """
        kernel_matrix = self.weight.view(self.out_channels, -1)
        # shape of out_h for tensor network contraction
        h_out_structure = self.h_out_tensor_structure()
        # perform tensor network contraction
        unfolded_hessian = einsum(
            "ij,ilmn,mp->jlpn",
            (kernel_matrix, out_h.view(h_out_structure), kernel_matrix),
        ).detach()

        # reshape into square matrix
        shape = 2 * (prod(self.mean_unfolded_input.size()),)
        return unfolded_hessian.contiguous().view(shape)

    def init_bias_hessian(self, output_hessian):
        """Initialized bias attributes hessian and hvp.

        Initializes:
        ------------
        self.bias.hessian: Holds a matrix representing the batch-averaged
                           Hessian with respect to the bias
        self.bias.hvp: Provides implicit matrix-vector multiplication
                       routine by the batch-averaged bias Hessian

        Parameters:
        -----------
        out_h (torch.Tensor): Batch-averaged Hessian with respect to
                              the layer's outputs
        """
        shape = self.h_out_tensor_structure()
        self.bias.hessian = output_hessian.view(shape).sum(3).sum(1)
        self.bias.hvp = self._bias_hessian_vp

    def init_weight_hessian(self, out_h):
        """Create weight attributes hessian and hvp.

        Initializes:
        ------------
        self.weight.hessian: Holds a function which, when called, returns
                             a matrix representing the batch-averaged
                             Hessian with respect to the weights
        self.weight.hvp: Provides implicit matrix-vector multiplication
                         routine by the batch-averaged weight Hessian

        Parameters:
        -----------
        out_h (torch.Tensor): Batch-averaged Hessian with respect to
                              the layer's outputs
        """
        self.weight.hessian = self._compute_weight_hessian(out_h)
        self.weight.hvp = self._weight_hessian_vp(out_h)

    def _bias_hessian_vp(self, v):
        """Matrix multiplication by bias Hessian.

        Parameters:
        -----------
        v (torch.Tensor): Vector which is to be multiplied by the Hessian

        Returns:
        --------
        result (torch.Tensor): bias_hessian * v
        """
        return self.bias.hessian.matmul(v)

    def _weight_hessian_vp(self, out_h):
        """Matrix multiplication by weight Hessian.
        """

        def hvp(v):
            r"""Matrix-vector product with weight Hessian.

            Use approximation
            weight_hessian = (I \otimes X) output_hessian (I \otimes X^T).

            Parameters:
            -----------
            v (torch.Tensor): Vector which is multiplied by the Hessian
            """
            if not len(v.size()) == 1:
                raise ValueError("Require one-dimensional tensor")
            # reshape vector into (out_channels, -1)
            temp = v.view(self.out_channels, -1)
            # perform tensor network contraction
            result = einsum(
                "kl,jlmp,op,mo->jk",
                (
                    self.mean_unfolded_input,
                    out_h.view(self.h_out_tensor_structure()),
                    self.mean_unfolded_input,
                    temp,
                ),
            )
            return result.view(v.size())

        return hvp

    def _compute_weight_hessian(self, out_h):
        r"""Compute weight Hessian from output Hessian.

        Use approximation
        weight_hessian = (I \otimes X) output_hessian (I \otimes X^T).
        """

        def weight_hessian():
            """Compute matrix form of the weight Hessian when called."""
            # compute the weight Hessian
            w_hessian = einsum(
                "kl,jlmp,op->jkmo",
                (
                    self.mean_unfolded_input,
                    out_h.view(self.h_out_tensor_structure()),
                    self.mean_unfolded_input,
                ),
            )
            # reshape into square matrix
            num_weight = self.weight.numel()
            return w_hessian.view(num_weight, num_weight)

        return weight_hessian

    def h_out_tensor_structure(self):
        """Return tensor shape of output Hessian for weight Hessian.

        The rank-4 shape is given by (out_channels, num_patches,
        out_channels, num_patches)."""
        num_patches = self.mean_unfolded_input.size()[1]
        return 2 * (self.out_channels, num_patches)

    @staticmethod
    def output_shape(
        input_size, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        """Compute the size of the output from a forward pass.

        Parameters:
        -----------
        input_size : tuple(int)
            4-dimensional tuple containing the dimensions of the input.

        The remaining parameters are the same as for ``Conv2d``.

        Returns:
        --------
        tuple(int) :
            Dimension of the output
        """
        if not len(input_size) == 4:
            raise ValueError(
                "Expecting 4-dimensional input, but got {} dimensions".format(
                    len(input_size)
                )
            )
        layer = Conv2d(
            in_channels=input_size[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        output = layer(zeros(*input_size))
        return tuple(output.size())

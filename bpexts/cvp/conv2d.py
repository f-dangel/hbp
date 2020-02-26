"""Curvature-vector products for linear layers."""

from numpy import prod
from torch import einsum
from torch.nn import Conv2d, functional

from ..hbp.module import hbp_decorate


class CVPConv2d(hbp_decorate(Conv2d)):
    """2D Convolution with recursive Hessian-vector products."""

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
    def hbp_hooks(self):
        """Install hook storing unfolded input."""
        self.register_exts_forward_hook(self.store_input_and_output_dimensions)

    # --- hooks ---
    @staticmethod
    def store_input_and_output_dimensions(module, input, output):
        """Save input and dimensions of the output to the layer.

        Intended use as forward hook.
        Initialize module buffer 'layer_input' and attribute
        'output_size'.
        """
        if not len(input) == 1:
            raise ValueError("Cannot handle multi-input scenario")
        layer_input = input[0].data  # detach()
        module.register_exts_buffer("layer_input", layer_input)
        module.output_size = tuple(output.size())

    # --- end of hooks ---

    # --- Hessian-vector product with the input Hessian ---
    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms="none"):
        """Return CVP with respect to the input."""

        def _input_hessian_vp(v):
            """Multiplication by the Hessian w.r.t. the input."""
            return self._input_jacobian_transpose(
                output_hessian(self._input_jacobian(v))
            )

        return _input_hessian_vp

    def _input_jacobian(self, v):
        """Apply the Jacobian with respect to the input."""
        batch, in_channels, in_x, in_y = tuple(self.layer_input.size())
        assert tuple(v.size()) == (self.layer_input.numel(),)
        result = v.view(batch, in_channels, in_x, in_y)
        result = functional.conv2d(
            result,
            self.weight.data,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        assert tuple(result.size()) == self.output_size
        return result.view(-1)

    def _input_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the input.

        Note
        ----
        The transpose Jacobian is implemented with ``conv_transpose2d``.
        For strides larger than one, this can currently lead to failing
        assertion statements. It is cause by different shapes being
        mapped to the same output dimension by conv/conv_transpose.

        Reference
        ---------
        https://github.com/keras-team/keras/issues/9883

        """
        batch, in_channels, in_x, in_y = tuple(self.layer_input.size())
        assert tuple(v.size()) == (prod(self.output_size),)
        result = v.view(*self.output_size)
        result = functional.conv_transpose2d(
            result,
            self.weight.data,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if tuple(result.size()) != tuple(self.layer_input.size()):
            raise ValueError(
                "Size after conv_transpose does not match",
                " the input size. This occurs for strides",
                " larger than 1. TO BE FIXED. Got {}, expected {}".format(
                    result.size(), self.layer_input.size()
                ),
            )

        return result.view(-1)

    # --- Hessian-vector products with the parameter Hessians ---
    # override
    def parameter_hessian(self, output_hessian):
        """Initialize VPs with the layer parameter Hessian."""
        if self.bias is not None:
            self.init_bias_hessian(output_hessian)
        self.init_weight_hessian(output_hessian)

    # --- bias term ---
    def init_bias_hessian(self, output_hessian):
        """Initialize bias Hessian-vector product."""

        def _bias_hessian_vp(v):
            """Multiplication by the bias Hessian."""
            return self._bias_jacobian_transpose(output_hessian(self._bias_jacobian(v)))

        self.bias.hvp = _bias_hessian_vp

    def _bias_jacobian(self, v):
        """Apply the Jacobian with respect to the bias."""
        assert tuple(v.size()) == (self.bias.numel(),)
        result = v.view(1, self.bias.numel(), 1, 1)
        result = result.expand(*self.output_size)
        assert tuple(result.size()) == self.output_size
        return result.contiguous().view(-1)

    def _bias_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the bias."""
        assert tuple(v.size()) == (prod(self.output_size),)
        result = v.view(*self.output_size).sum(3).sum(2).sum(0)
        assert tuple(result.size()) == (self.bias.numel(),)
        return result

    def init_weight_hessian(self, output_hessian):
        """Initialize weight Hessian-vector product."""

        def _weight_hessian_vp(v):
            """Multiplication by the weight Hessian."""
            return self._weight_jacobian_transpose(
                output_hessian(self._weight_jacobian(v))
            )

        self.weight.hvp = _weight_hessian_vp

    def _weight_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the weights."""
        batch, out_channels, out_x, out_y = self.output_size
        assert tuple(v.size()) == (prod(self.output_size),)

        _, in_channels, in_x, in_y = self.layer_input.size()
        k_x, k_y = self.kernel_size

        result = v.view(batch, out_channels, out_x, out_y)
        result = result.repeat(1, in_channels, 1, 1)
        result = result.view(batch * out_channels * in_channels, 1, out_x, out_y)

        input = self.layer_input.view(1, -1, in_x, in_y)

        result = functional.conv2d(
            input,
            result,
            None,
            self.dilation,
            self.padding,
            self.stride,
            in_channels * batch,
        )

        result = result.view(batch, out_channels * in_channels, k_x, k_y)
        result = result.sum(0).view(in_channels, out_channels, k_x, k_y)
        result = einsum("mnxy->nmxy", result).contiguous().view(-1)
        assert tuple(result.size()) == (self.weight.numel(),)

        return result

    def _weight_jacobian(self, v):
        """Apply the Jacobian with respect to the weight."""
        batch, in_channels, in_x, in_y = tuple(self.layer_input.size())
        assert tuple(v.size()) == (self.weight.numel(),)
        result = v.view_as(self.weight)

        result = functional.conv2d(
            self.layer_input,
            result,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        assert tuple(result.size()) == self.output_size
        return result.view(-1)

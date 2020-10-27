import torch
import torch.nn as nn
import torch.nn.functional as F

class LowerBound_alt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class GDN_alt(nn.Module):
    """
    Generalized divisive normalization layer.
    Based on the papers:
    "Density modeling of images using a generalized normalization transformation"
    J. Ballé, V. Laparra, E.P. Simoncelli
    https://arxiv.org/abs/1511.06281
    "End-to-end optimized image compression"
    J. Ballé, V. Laparra, E.P. Simoncelli
    https://arxiv.org/abs/1611.01704
    Implements an activation function that is essentially a multivariate
    generalization of a particular sigmoid-type function:
    ```
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    ```
    where `i` and `j` run over channels. This implementation never sums across
    spatial dimensions. It is similar to local response normalization, but much
    more flexible, as `beta` and `gamma` are trainable parameters.
    Based on tensorflow.github.io/compression/docs/api_docs/python/tfc/GDN.html
             github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/parameterizers.py
    """

    def __init__(self, n_channels, inverse=False, rectify=False, gamma_init=0.1, beta_min=1e-6, 
                 reparam_offset=2**(-18), **kwargs):
        super(GDN, self).__init__()

        self.n_channels = n_channels
        self.inverse = inverse
        self.rectify = rectify
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        # Initialize trainable affine params
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**0.5
        self.gamma_bound = self.reparam_offset

        self.beta = nn.Parameter(torch.sqrt(torch.ones(n_channels) + self.pedestal))
        self.gamma = nn.Parameter(torch.sqrt(self.gamma_init * torch.eye(n_channels) + self.pedestal))


    def forward(self, x):
        # Expects input dimensions (B,C,H,W)

        n_channels = x.size()[1]

        # stored variables parameterized as square roots
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(n_channels, n_channels, 1, 1)

        norm_pool = F.conv2d(x**2, gamma, bias=beta)

        if self.inverse is True:
            # Approximate inverse by one round of fixed-point iteration
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)

        return x * norm_pool


class GDN1_alt(GDN_alt):
    """
    Simplified GDN layer.
    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston, Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).
    .. math::
        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}
    """

    def forward(self, x):
        # Expects input dimensions (B,C,H,W)

        n_channels = x.size()[1]

        # stored variables parameterized as square roots
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(n_channels, n_channels, 1, 1)

        norm_pool = F.conv2d(torch.abs(x), gamma, bias=beta)

        if not self.inverse:
            norm_pool = 1.0 / norm_pool

        return x * norm_pool

class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.
    Used for stability during training.
    """

    def __init__(self, minimum=0, reparam_offset=2 ** -18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset ** 2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x):
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """

    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class GDN1(GDN):
    r"""Simplified GDN layer.
    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston, Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).
    .. math::
        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}
    """

    def forward(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# Custom
from src.helpers import maths, utils

MIN_SCALE = 0.11
MIN_LIKELIHOOD = 1e-9
TAIL_MASS = 2**-8
PRECISION_P = 24  # Precision of rANS coder

HyperInfo = namedtuple(
    "HyperInfo",
    "decoded "
    "latent_nbpp hyperlatent_nbpp total_nbpp latent_qbpp hyperlatent_qbpp total_qbpp "
    "bitstring side_bitstring",
)

lower_bound_toward = maths.LowerBoundToward.apply

# TODO unit test on simple CDFs
def estimate_tails(cdf, target, shape, dtype=torch.float32, extra_counts=10):
    """
    Estimates approximate tail quantiles.
    This runs a simple Adam iteration to determine tail quantiles. The
    objective is to find an `x` such that: [[[ cdf(x) == target ]]]

    Note that `cdf` is assumed to be monotonic. When each tail estimate has passed the 
    optimal value of `x`, the algorithm does `extra_counts` (default 10) additional 
    iterations and then stops.

    This operation is vectorized. The tensor shape of `x` is given by `shape`, and
    `target` must have a shape that is broadcastable to the output of `func(x)`.

    Arguments:
    cdf: A callable that computes cumulative distribution function, survival
         function, or similar.
    target: The desired target value.
    shape: The shape of the `tf.Tensor` representing `x`.
    Returns:
    A `torch.Tensor` representing the solution (`x`).
    """
    # bit hacky
    lr, eps = 1e-2, 1e-8
    beta_1, beta_2 = 0.9, 0.99

    # Tails should be monotonically increasing
    tails = torch.zeros(shape, dtype=dtype, requires_grad=True)
    m = torch.zeros(shape, dtype=dtype)
    v = torch.ones(shape, dtype=dtype)
    counts = torch.zeros(shape, dtype=torch.int32)

    while torch.min(counts) < extra_counts:
        loss = abs(cdf(tails) - target)
        loss.backward(torch.ones_like(tails))

        grad = tails.grad

        with torch.no_grad():
            m = beta_1 * m + (1. - beta_1) * tails.grad
            v = beta_2 * v + (1. - beta_2) * torch.square(tails.grad)
            tails -= lr * m / (torch.sqrt(v) + eps)

        # Condition assumes tails init'd at zero
        counts = torch.where(torch.logical_or(counts > 0, tails.grad * tails > 0), 
            counts+1, counts)

        tails.grad.zero_()

    return tails


class ContinuousEntropyModel(nn.Module):
    """
    Pre-compute integer probability tables for use in rANS 
    encoder/decoder
    """
    def __init__(self, distribution, likelihood_bound=MIN_LIKELIHOOD, tail_mass=TAIL_MASS,
        precision=PRECISION_P):
        """
        Parameters:
            distribution: Distribution with CDF / quantile / likelihood methods
        """

        self.distribution = distribution
        self.likelihood_bound = likelihood_bound
        self.tail_mass = tail_mass
        self.precision = precision

        self.build_tables()

    def quantize_st(self, inputs, offsets=None):
        # Ignore rounding in backward pass
        values = inputs

        if offsets is not None:
            values = values - offsets

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if offsets is not None:
            values = values + offsets

        return values

    def build_tables(self, mean=None, scale=None):

        offsets = mean
        if offsets is None:
            offsets = 0.
        
        lower_tail = self.distribution.lower_tail(self.tail_mass)
        upper_tail = self.distribution.upper_tail(self.tail_mass)

        # Largest distance observed between lower tail and median, 
        # and between median and upper tail.
        minima = offsets - lower_tail
        minima = torch.ceil(minima).to(torch.int32)
        minima = torch.max(minima, 0)[0]

        maxima = upper_tail - offsets
        maxima = torch.ceil(maxima).to(torch.int32)
        maxima = torch.max(maxima, 0)[0]

        # PMF starting positions and lengths
        # pmf_start = offsets - minima.to(self.distribution.dtype)
        pmf_start = offsets - minima.to(torch.float32)
        pmf_length = maxima + minima + 1  # Symmetric for Gaussian

        max_length = torch.max(pmf_length)
        samples = torch.arange(max_length, dtype=self.distribution.dtype)

        samples = torch.reshape(samples, (-1, self.distribution.n_channels))
        samples += pmf_start 
        pmf = self.distribution.likelihood(samples, mean=mean, scale=scale)

        pmf = torch.reshape(pmf, (max_length, -1))
        pmf = torch.transpose(pmf, 0, 1)

        

class PriorDensity(nn.Module):
    """
    Probability model for latents y. Based on Sec. 3. of [1].
    Returns convolution of Gaussian / logistic latent density with parameterized 
    mean and variance with uniform distribution U(-1/2, 1/2).

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).
    """

    def __init__(self, n_channels, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD,
        scale_lower_bound=MIN_SCALE, likelihood_type='gaussian', **kwargs):
        super(PriorDensity, self).__init__()

        self.n_channels = n_channels
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)
        self.scale_lower_bound = scale_lower_bound
        self.likelihood_type = likelihood_type
        self.dtype = torch.float32

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
            self.standardized_quantile = maths.standardized_quantile_gaussian

        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
            self.standardized_quantile = maths.standardized_quantile_logistic

    def quantization_offset(self, mean, **kwargs):
        """
        x_quantized = torch.round(x - offset) + offset
        Where `offset` is gradient-less
        """
        return mean.detach()

    def lower_tail(self, tail_mass):
        lt = self.standardized_quantile(0.5 * tail_mass)
        return torch.ones(self.n_channels) * lt

    def upper_tail(self, tail_mass):
        ut = self.standardized_quantile(1 - 0.5 * tail_mass)
        return torch.ones(self.n_channels) * ut

    def likelihood(self, x, mean, scale):

        # Assumes 1 - CDF(x) = CDF(-x)
        x = x - mean
        x = torch.abs(x)
        cdf_upper = self.standardized_CDF((0.5 - x) / scale)
        cdf_lower = self.standardized_CDF(-(0.5 + x) / scale)

        likelihood_ = cdf_upper - cdf_lower
        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        return likelihood_

    def forward(self, x, mean, scale, **kwargs):
        return self.likelihood(x, mean, scale)

class HyperpriorDensity(nn.Module):
    """
    Probability model for hyper-latents z. Based on Sec. 6.1. of [1].
    Returns convolution of non-parametric hyperlatent density with uniform distribution 
    U(-1/2, 1/2).

    Assumes that the input tensor is at least 2D, with a batch dimension
    at the beginning and a channel dimension as specified by `data_format`. The
    layer trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the inputs are i.i.d.

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).
    """

    def __init__(self, n_channels, init_scale=10., filters=(3, 3, 3), min_likelihood=MIN_LIKELIHOOD, 
        max_likelihood=MAX_LIKELIHOOD, **kwargs):
        """
        init_scale: Scaling factor determining the initial width of the
                    probability densities.
        filters:    Number of filters at each layer < K
                    of the density model. Default K=4 layers.
        """
        super(HyperpriorDensity, self).__init__()
        
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)
        self.n_channels = n_channels
        self.dtype = torch.float32

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Define univariate density model 
        for k in range(len(self.filters)+1):
            
            # Weights
            H_init = np.log(np.expm1(1 / scale / filters[k + 1]))
            H_k = nn.Parameter(torch.ones((n_channels, filters[k+1], filters[k])))  # apply softmax for non-negativity
            torch.nn.init.constant_(H_k, H_init)
            self.register_parameter('H_{}'.format(k), H_k)

            # Scale factors
            a_k = nn.Parameter(torch.zeros((n_channels, filters[k+1], 1)))
            self.register_parameter('a_{}'.format(k), a_k)

            # Biases
            b_k = nn.Parameter(torch.zeros((n_channels, filters[k+1], 1)))
            torch.nn.init.uniform_(b_k, -0.5, 0.5)
            self.register_parameter('b_{}'.format(k), b_k)

    def cdf_logits(self, x, update_parameters=True):
        """
        Evaluate logits of the cumulative densities. 
        Independent density model for each channel.

        x:  The values at which to evaluate the cumulative densities.
            torch.Tensor - shape `(C, 1, *)`.
        """
        logits = x

        for k in range(len(self.filters)+1):
            H_k = getattr(self, 'H_{}'.format(str(k)))  # Weight
            a_k = getattr(self, 'a_{}'.format(str(k)))  # Scale
            b_k = getattr(self, 'b_{}'.format(str(k)))  # Bias

            if update_parameters is False:
                H_k, a_k, b_k = H_k.detach(), a_k.detach(), b_k.detach()
            logits = torch.bmm(F.softplus(H_k), logits)  # [C,filters[k+1],*]
            logits = logits + b_k
            logits = logits + torch.tanh(a_k) * torch.tanh(logits)

        return logits

    def quantization_offset(self, **kwargs):
        return 0.

    def lower_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        lt = estimate_tails(cdf_logits_func, target=-torch.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()

        return lt.reshape(self.n_channels)

    def upper_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        ut = estimate_tails(cdf_logits_func, target=torch.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()

        return ut.reshape(self.n_channels)

    def likelihood(self, x, **kwargs):
        """
        Expected input: (N,C,H,W)
        """
        latents = x

        # Converts latents to (C,1,*) format
        N, C, H, W = latents.size()
        latents = latents.permute(1,0,2,3)
        shape = latents.shape
        latents = torch.reshape(latents, (shape[0],1,-1))

        cdf_upper = self.cdf_logits(latents + 0.5)
        cdf_lower = self.cdf_logits(latents - 0.5)

        # Numerical stability using some sigmoid identities
        # to avoid subtraction of two numbers close to 1
        sign = -torch.sign(cdf_upper + cdf_lower)
        sign = sign.detach()
        likelihood_ = torch.abs(
            torch.sigmoid(sign * cdf_upper) - torch.sigmoid(sign * cdf_lower))

        # Naive
        # likelihood_ = torch.sigmoid(cdf_upper) - torch.sigmoid(cdf_lower)

        # Reshape to (N,C,H,W)
        likelihood_ = torch.reshape(likelihood_, shape)
        likelihood_ = likelihood_.permute(1,0,2,3)

        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        return likelihood_

    def forward(self, x, **kwargs):
        return self.likelihood(x)



if __name__ == '__main__':

    quantile = 0.42

    norm_cdf = lambda x: 0.5 * (1. + torch.erf(x / np.sqrt(2)))
    tails = estimate_tails(norm_cdf, quantile, shape=10)

    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")

    quantile = 0.87

    norm_cdf = torch.sigmoid
    tails = estimate_tails(norm_cdf, quantile, shape=10)

    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")
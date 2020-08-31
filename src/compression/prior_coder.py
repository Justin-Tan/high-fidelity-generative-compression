import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom
from src.helpers import maths
from src.compression import entropy_models

lower_bound_toward = maths.LowerBoundToward.apply

MIN_SCALE = entropy_models.MIN_SCALE
MIN_LIKELIHOOD = entropy_models.MIN_LIKELIHOOD
TAIL_MASS = entropy_models.TAIL_MASS
PRECISION_P = entropy_models.PRECISION_P

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def prior_scale_table(scales_min=SCALES_MIN, scales_max=SCALES_MAX, levels=SCALES_LEVELS):
    scale_table = np.exp(np.linspace(np.log(scales_min), np.log(scales_max), levels))
    return scale_table


class PriorEntropyModel(entropy_models.ContinuousEntropyModel):

    """
    Routines for compression/decompression using prior entropy model.

    This class assumes that all scalar elements of the encoded tensor are
    conditionally independent given some other random variable, possibly depending
    on data. All dependencies must be represented by the `indexes` tensor. For
    each bottleneck tensor element, it selects the appropriate scalar
    distribution.

    """

    def __init__(self, distribution, scale_table=None, index_ranges=64, min_scale=MIN_SCALE,
        likelihood_bound=MIN_LIKELIHOOD, tail_mass=TAIL_MASS, precision=PRECISION_P):

        """
        `scale_table`: Iterable of positive floats. For range coding, the scale
        parameters in `scale` can't be used, because the probability tables need
        to be constructed statically. Only the values given in this table will
        actually be used for range coding. For each predicted scale, the next
        greater entry in the table is selected. It's optimal to choose the
        scales provided here in a logarithmic way.
        """

        self.scale_table = scale_table
        self.index_ranges = int(index_ranges)
        self.min_scale = min_scale

        if self.scale_table is None:
            self.scale_table = prior_scale_table()

        self.scale_table = lower_bound_toward(self.scale_table, self.min_scale)
        self.indices = torch.arange(self.index_ranges)   

        scale_table_tensor = torch.Tensor(tuple(float(s) for s in self.scale_table))
        self.register_buffer('scale_table', self._prepare_scale_table(scale_table_tensor))
        self.register_buffer('min_scale', torch.Tensor([float(self.min_scale)]))

        super().__init__(distribution=distribution, likelihood_bound=likelihood_bound, 
            tail_mass=tail_mass, precision=precision)
  
    def build_tables(self, **kwargs):

        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = np.ceil(np.array(self.scale_table) * multiplier).astype(int)
        pmf_length = 2 * pmf_center + 1
        # [n_scales]
        max_length = np.max(pmf_length)

        samples = torch.abs(torch.arange(max_length).int() - pmf_center[:, None]).float()
        samples_scale = self.scale_table.unsqueeze(1).float()

        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower  # [n_scales, max_length]

        # [n_scales]
        tail_mass = 2 * lower[:, :1]
        cdf_length = pmf_length + 2
        cdf_offset = -pmf_center

        # CDF shape [n_scales,  max_length + 2] - account for fenceposts + overflow
        CDF = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for n, (pmf_, pmf_length_, tail_) in enumerate(zip(pmf, pmf_length, tail_mass)):
            pmf_ = pmf_[:pmf_length_]  # [max_length]
            overflow = torch.clamp(1. - torch.sum(pmf_, keepdim=True), min=0.)
            # pmf_ = torch.cat((pmf_, overflow), dim=0)
            pmf_ = torch.cat((pmf_, tail_), dim=0)

            cdf_ = maths.pmf_to_quantized_cdf(pmf_, self.precision)
            cdf_ = F.pad(cdf, (0, max_length - pmf_length_), mode='constant', value=0)
            CDF[n] = cdf_

        # Serialize, compression method responsible for identifying which 
        # CDF to use during compression
        self.CDF = nn.Parameter(CDF, requires_grad=False)
        self.CDF_offset = nn.Parameter(cdf_offset, requires_grad=False)
        self.CDF_length = nn.Parameter(cdf_length, requires_grad=False)

        self.register_parameter('CDF', self.CDF)
        self.register_parameter('CDF_offset', self.CDF_offset)
        self.register_parameter('CDF_length', self.CDF_length)

    def _estimate_compression_bits(self, x, means, scales, spatial_shape):
        """
        Estimate number of bits needed to compress `x`
        Assumes each channel is compressed to its own bit string.

        Parameters:
            x:              Bottleneck tensor to be compressed, [N,C,H,W]
            spatial_shape:  Spatial dimensions of original image
        """

        EPS = 1e-9
        quotient = -np.log(2.)
        quantized = self.quantize_st(x, offsets=means)
        likelihood = self.distribution.likelihood(quantized, means, scales)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def compute_indices(self, scales):
        # Compute the indexes into the table for each of the passed-in scales.
        scales = lower_bound_toward(scales)
        indices = torch.ones_like(scales, dtype=torch.int32) * (len(self.scale_table) - 1)

        for s in range(scale_table[:-1]):
            indices = indices - (scales <= s).to(torch.int32) 

        return indices      

    def compress(self, bottleneck, means, scales):
        """
        Compresses floating-point tensors to bitsrings.

        Compresses the tensor to bit strings. `bottleneck` is first quantized
        as in `quantize()`, and then compressed using the probability tables derived
        from the entropy model. The quantized tensor can later be recovered by
        calling `decompress()`.

        Arguments:
        bottleneck: Data to be compressed. Format (N,C,H,W).
        Returns:
        strings:    Tensor of the same shape as `bottleneck` containing a string for 
                    each batch entry.
        """
        input_shape = tuple(bottleneck.size())
        batch_shape = input_shape[0]
        coding_shape = input_shape[1:]

        indices = self.compute_indices(scales)
        symbols = torch.round(bottleneck - means).to(torch.int32)

        assert symbols.size() == indices.size(), 'Indices should have same size as inputs.'
        assert len(symbols.size() == 4), 'Expect (N,C,H,W)-format input.'

        strings = torch.zeros(batch_shape)

        for i in range(symbols.size(0)):

            coded_string = entropy_coding.ans_index_encode(
                symbols=symbols[i],
                indices=indices,
                cdf=self.CDF,
                cdf_length=self.CDF_length,
                cdf_offset=self.CDF_offset,
                precision=self.precision,
            )

            strings[i] = coded_string

        return strings

    
    def decompress(self, strings, means, scales, broadcast_shape):
        """
        Decompress bitstrings to floating-point tensors.

        Reconstructs the quantized tensor from bitstrings produced by `compress()`.
        It is necessary to provide a part of the output shape in `broadcast_shape`.

        Arguments:
        strings:            Tensor containing the compressed bit strings.   

        Returns:
        reconstruction:     Tensor of same shape as input to `compress()`.
        """

        batch_shape = strings.size(0)
        n_channels = self.distribution.n_channels
        # same as `input_shape` to `compress()`
        symbols_shape =  (batch_shape, n_channels, *broadcast_shape)

        indices = self.compute_indices(scales)
        strings = torch.reshape(strings, (-1,))

        assert len(indices.size() == 4), 'Expect (N,C,H,W)-format input.'
        assert strings.size(0) == indices.size(0), 'Batch shape mismatch!'
        assert means.size(1) == indices.size(1), 'Mean dims mismatch!'
        if means.size() != indices.size():
            if ((means.size(2) != 1) and (means.size(3) != 1)):
                raise ValueError('Mean dims mismatch!')

        symbols = torch.zeros(symbols_shape, dtype=torch.float32)

        for i in range(strings.size(0)):

            decoded = entropy_coder.ans_index_decode(
                bitstring=strings[i],
                indices=indices,
                cdf=self.CDF,
                cdf_length=self.CDF_length,
                cdf_offset=self.CDF_offset,
                precision=self.precision,
            )

            symbols[i] = decoded

        symbols = torch.reshape(symbols, symbols_shape)
        outputs = self.dequantize(symbols, offsets=means)

        return outputs



class PriorDensity(nn.Module):
    """
    Probability model for latents y. Based on Sec. 3. of [1].
    Returns convolution of Gaussian / logistic latent density with parameterized 
    mean and variance with 'boxcar' uniform distribution U(-1/2, 1/2).

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
            self.quantile = maths.quantile_gaussian

        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
            self.standardized_quantile = maths.standardized_quantile_logistic
            self.quantile = maths.quantile_logistic

    def quantization_offset(self, mean, **kwargs):
        """
        x_quantized = torch.round(x - offset) + offset
        Where `offset` is gradient-less
        """
        return mean.detach()

    def lower_tail(self, tail_mass, mean, scale):
        tail_mass = float(tail_mass)
        lt = self.quantile(0.5 * tail_mass, mean=mean, scale=scale)
        return lt

    def upper_tail(self, tail_mass, mean, scale):
        tail_mass = float(tail_mass)
        ut = self.quantile(1. - 0.5 * tail_mass, mean=mean, scale=scale)
        return ut

    def likelihood(self, x, mean, scale, **kwargs):

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
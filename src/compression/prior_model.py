import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Custom
from src.helpers import maths
from src.compression import entropy_models, entropy_coding
from src.compression import compression_utils

lower_bound_toward = maths.LowerBoundToward.apply

MIN_SCALE = entropy_models.MIN_SCALE
MIN_LIKELIHOOD = entropy_models.MIN_LIKELIHOOD
MAX_LIKELIHOOD = entropy_models.MAX_LIKELIHOOD
TAIL_MASS = entropy_models.TAIL_MASS
PRECISION_P = entropy_models.PRECISION_P

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def prior_scale_table(scales_min=SCALES_MIN, scales_max=SCALES_MAX, levels=SCALES_LEVELS):
    scale_table = np.exp(np.linspace(np.log(scales_min), np.log(scales_max), levels))
    return torch.Tensor(scale_table)


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

        self.standardized_CDF = distribution.standardized_CDF
        self.standardized_quantile = distribution.standardized_quantile
        self.quantile = distribution.quantile

        super().__init__(distribution=distribution, likelihood_bound=likelihood_bound, 
            tail_mass=tail_mass, precision=precision)

        self.build_tables()
        scale_table_tensor = torch.Tensor(tuple(float(s) for s in self.scale_table))
        self.register_buffer('scale_table_tensor', scale_table_tensor)
        self.register_buffer('min_scale_tensor', torch.Tensor([float(self.min_scale)]))


    def build_tables(self, **kwargs):

        multiplier = -self.standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).to(torch.int32)
        pmf_length = 2 * pmf_center + 1  # [n_scales]
        max_length = torch.max(pmf_length).item()

        samples = torch.abs(torch.arange(max_length).int() - pmf_center[:, None]).float()
        samples_scale = self.scale_table.unsqueeze(1).float()

        upper = self.standardized_CDF((.5 - samples) / samples_scale)
        lower = self.standardized_CDF((-.5 - samples) / samples_scale)
        pmf = upper - lower  # [n_scales, max_length]

        # [n_scales]
        tail_mass = 2 * lower[:, :1]
        cdf_offset = -pmf_center  # How far away we have to go to pass the tail mass
        cdf_length = pmf_length + 2
        cdf_length = cdf_length.to(torch.int32)
        cdf_offset = cdf_offset.to(torch.int32)

        # CDF shape [n_scales,  max_length + 2] - account for fenceposts + overflow
        CDF = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for n, (pmf_, pmf_length_, tail_) in enumerate((zip(tqdm(pmf), pmf_length, tail_mass))): 
            pmf_ = pmf_[:pmf_length_]  # [max_length]
            overflow = torch.clamp(1. - torch.sum(pmf_, dim=0, keepdim=True), min=0.)  
            # pmf_ = torch.cat((pmf_, overflow), dim=0)
            pmf_ = torch.cat((pmf_, tail_), dim=0)

            cdf_ = maths.pmf_to_quantized_cdf(pmf_, self.precision)
            cdf_ = F.pad(cdf_, (0, max_length - pmf_length_), mode='constant', value=0)
            CDF[n] = cdf_

        # Serialize, compression method responsible for identifying which 
        # CDF to use during compression
        self.CDF = nn.Parameter(CDF, requires_grad=False)  # Last entry in CDF is overflow code.
        self.CDF_offset = nn.Parameter(cdf_offset, requires_grad=False)
        self.CDF_length = nn.Parameter(cdf_length, requires_grad=False)

        compression_utils.check_argument_shapes(self.CDF, self.CDF_length, self.CDF_offset)

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
        n_bits = torch.sum(log_likelihood) / quotient
        bpi = n_bits / batch_size
        bpp = n_bits / n_pixels

        return n_bits, bpp, bpi

    def compute_indices(self, scales):
        # Compute the indexes into the table for each of the passed-in scales.
        scales = lower_bound_toward(scales, SCALES_MIN)
        indices = torch.ones_like(scales, dtype=torch.int32) * (len(self.scale_table) - 1)

        for s in self.scale_table[:-1]:
            indices = indices - (scales <= s).to(torch.int32) 

        return indices      

    def compress(self, bottleneck, means, scales, vectorize=False, block_encode=True):
        """
        Compresses floating-point tensors to bitsrings.

        Compresses the tensor to bit strings. `bottleneck` is first quantized
        as in `quantize()`, and then compressed using the probability tables derived
        from the entropy model. The quantized tensor can later be recovered by
        calling `decompress()`.

        Arguments:
        bottleneck:     Data to be compressed. Format (N,C,H,W).
        means:          Format (N,C,H,W).
        scales:         Format (N,C,H,W).

        Returns:
        encoded:        Tensor of the same shape as `bottleneck` containing the 
                        compressed message.
        """
        input_shape = tuple(bottleneck.size())
        batch_shape = input_shape[0]
        coding_shape = input_shape[1:]  # (C,H,W)

        indices = self.compute_indices(scales)
        symbols = torch.floor(bottleneck + 0.5 - means).to(torch.int32)
        rounded = symbols.clone()
        # symbols = torch.round(bottleneck - means).to(torch.int32)

        assert symbols.size() == indices.size(), 'Indices should have same size as inputs.'
        assert len(symbols.size()) == 4, 'Expect (N,C,H,W)-format input.'

        # All inputs should be integer-typed
        symbols = symbols.cpu().numpy()
        indices = indices.cpu().numpy()

        cdf = self.CDF.cpu().numpy().astype('uint32')
        cdf_length = self.CDF_length.cpu().numpy()
        cdf_offset = self.CDF_offset.cpu().numpy()

        encoded, coding_shape = compression_utils.ans_compress(symbols, indices, cdf, cdf_length, cdf_offset,
            coding_shape, precision=self.precision, vectorize=vectorize, 
            block_encode=block_encode)

        return encoded, coding_shape, rounded

    
    def decompress(self, encoded, means, scales, broadcast_shape, coding_shape, vectorize=False, 
        block_decode=True):
        """
        Decompress bitstrings to floating-point tensors.

        Reconstructs the quantized tensor from bitstrings produced by `compress()`.
        It is necessary to provide a part of the output shape in `broadcast_shape`.

        Arguments:
        encoded:            Tensor containing compressed bit strings produced by
                            the `compress()` method. Arguments must be identical.
        broadcast_shape:    Iterable of ints. Spatial extent of quantized feature map. 
        coding_shape:       Shape of encoded messages.

        Returns:
        decoded:            Tensor of same shape as input to `compress()`.
        """

        batch_shape = scales.shape[0]
        n_channels = self.distribution.n_channels
        # same as `input_shape` to `compress()`
        symbols_shape = (batch_shape, n_channels, *broadcast_shape)

        indices = self.compute_indices(scales)

        assert len(indices.size()) == 4, 'Expect (N,C,H,W)-format input.'
        assert symbols_shape == indices.size(), 'Invalid indices!'
        assert means.size(1) == indices.size(1), 'Mean dims mismatch!'
        if means.size() != indices.size():
            if ((means.size(2) != 1) and (means.size(3) != 1)):
                raise ValueError('Mean dims mismatch!')

        indices = indices.cpu().numpy()
        cdf = self.CDF.cpu().numpy().astype('uint32')
        cdf_length = self.CDF_length.cpu().numpy()
        cdf_offset = self.CDF_offset.cpu().numpy()

        decoded = compression_utils.ans_decompress(encoded, indices, cdf, cdf_length, cdf_offset,
            coding_shape, precision=self.precision, vectorize=vectorize, block_decode=block_decode)

        symbols = torch.Tensor(decoded)
        symbols = torch.reshape(symbols, symbols_shape)
        decoded_raw = symbols.clone()
        decoded = self.dequantize(symbols, offsets=means)

        return decoded, decoded_raw



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

    
if __name__ == '__main__':

    import time

    n_channels = 24
    use_blocks = True
    vectorize = True
    prior_density = PriorDensity(n_channels)
    prior_entropy_model = PriorEntropyModel(distribution=prior_density)

    loc, scale = 2.401, 3.43
    n_data = 1
    toy_shape = (n_data, n_channels, 64, 64)
    bottleneck, means = torch.randn(toy_shape), torch.randn(toy_shape)
    scales = torch.randn(toy_shape) * np.sqrt(scale) + loc
    scales = torch.clamp(scales, min=MIN_SCALE)

    bits, bpp, bpi = prior_entropy_model._estimate_compression_bits(bottleneck, means, 
        scales, spatial_shape=toy_shape[2:])

    start_t = time.time()

    encoded, coding_shape, rounded = prior_entropy_model.compress(bottleneck, means, scales,
        block_encode=use_blocks, vectorize=vectorize)
    
    if (use_blocks is True) or (vectorize is True): 
        enc_shape = encoded.shape[0]
    else:
        enc_shape = sum([len(enc) for enc in encoded])

    print('Encoded shape', enc_shape)

    decoded, decoded_raw = prior_entropy_model.decompress(encoded, means, scales, 
        broadcast_shape=toy_shape[2:], coding_shape=coding_shape, block_decode=use_blocks,
        vectorize=vectorize)

    print('Decoded shape', decoded.shape)
    delta_t = time.time() - start_t
    print(f'Delta t {delta_t:.2f} s | ', torch.mean((decoded_raw == rounded).float()).item())

    cbits = enc_shape * 32
    print(f'Symbols compressed to {cbits:.1f} bits.')
    print(f'Estimated entropy {bits:.3f} bits.')

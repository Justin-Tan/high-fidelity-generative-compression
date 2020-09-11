import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Custom
from src.helpers import maths, utils
from src.compression import entropy_models, entropy_coding
from src.compression import compression_utils

MIN_SCALE = entropy_models.MIN_SCALE
MIN_LIKELIHOOD = entropy_models.MIN_LIKELIHOOD
MAX_LIKELIHOOD = entropy_models.MAX_LIKELIHOOD
TAIL_MASS = entropy_models.TAIL_MASS
PRECISION_P = entropy_models.PRECISION_P

lower_bound_toward = maths.LowerBoundToward.apply


class HyperpriorEntropyModel(entropy_models.ContinuousEntropyModel):

    """
    Routines for compression/decompression using hyperprior entropy model.

    This class assumes that all scalar elements of the encoded tensor are
    statistically independent, and that the parameters of their scalar
    distributions do not depend on data.

    """

    def __init__(self, distribution, likelihood_bound=MIN_LIKELIHOOD, tail_mass=TAIL_MASS,
        precision=PRECISION_P):

        super().__init__(distribution=distribution, likelihood_bound=likelihood_bound, 
            tail_mass=tail_mass, precision=precision)


    def compute_medians(self):
        self.medians = self.distribution.median().view(1,-1,1,1).cpu()

    def build_tables(self, **kwargs):
        
        offsets = 0.

        lower_tail = self.distribution.lower_tail(self.tail_mass).cpu()
        upper_tail = self.distribution.upper_tail(self.tail_mass).cpu()

        self.compute_medians()
        medians = torch.squeeze(self.medians)

        # Largest distance observed between lower tail and median, 
        # and between median and upper tail.
        minima = offsets - lower_tail
        minima = torch.ceil(minima).to(torch.int32)
        minima = torch.clamp(minima, min=0)

        maxima = upper_tail - offsets
        maxima = torch.ceil(maxima).to(torch.int32)
        maxima = torch.clamp(maxima, min=0)

        # PMF starting positions and lengths
        # pmf_start = offsets - minima.to(self.distribution.dtype)
        pmf_start = offsets - minima.to(torch.float32)
        pmf_length = maxima + minima + 1  # Symmetric for Gaussian

        max_length = pmf_length.max()
        samples = torch.arange(max_length, dtype=self.distribution.dtype)

        # Broadcast to [n_channels,1,*] format
        device = utils.get_device()
        samples = samples.view(1,-1) + pmf_start.view(-1,1,1)
        pmf = self.distribution.likelihood(samples.to(device), collapsed_format=True).cpu()

        # [n_channels, max_length]
        pmf = torch.squeeze(pmf)

        cdf_length = pmf_length + 2
        cdf_offset = -minima

        cdf_length = cdf_length.to(torch.int32)
        cdf_offset = cdf_offset.to(torch.int32)
        
        # CDF shape [n_channels, max_length + 2] - account for fenceposts + overflow
        CDF = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for n, (pmf_, pmf_length_) in enumerate(zip(tqdm(pmf), pmf_length)):
            pmf_ = pmf_[:pmf_length_]  # [max_length]
            overflow = torch.clamp(1. - torch.sum(pmf_, dim=0, keepdim=True), min=0.)
            pmf_ = torch.cat((pmf_, overflow), dim=0)

            cdf_ = maths.pmf_to_quantized_cdf(pmf_, self.precision)
            cdf_ = F.pad(cdf_, (0, max_length - pmf_length_), mode='constant', value=0)
            CDF[n] = cdf_

        # Serialize, compression method responsible for identifying which 
        # CDF to use during compression
        self.CDF = nn.Parameter(CDF, requires_grad=False)
        self.CDF_offset = nn.Parameter(cdf_offset, requires_grad=False)
        self.CDF_length = nn.Parameter(cdf_length, requires_grad=False)

        compression_utils.check_argument_shapes(self.CDF, self.CDF_length, self.CDF_offset)

        self.register_parameter('CDF', self.CDF)
        self.register_parameter('CDF_offset', self.CDF_offset)
        self.register_parameter('CDF_length', self.CDF_length)


    def _estimate_compression_bits(self, x, spatial_shape):
        """
        Estimate number of bits needed to compress `x`
        Assumes each channel is compressed to its own bit string.

        Parameters:
            x:              Bottleneck tensor to be compressed, [N,C,H,W]
            spatial_shape:  Spatial dimensions of original image
        """

        EPS = 1e-9
        quotient = -np.log(2.)

        quantized = self.quantize_st(x)
        likelihood = self.distribution.likelihood(quantized)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (quotient)
        bpi = n_bits / batch_size
        bpp = n_bits / n_pixels

        return n_bits, bpp, bpi

    def compute_indices(self, broadcast_shape):
        index_size = self.distribution.n_channels
        indices = torch.arange(index_size, dtype=torch.int32).view(-1, 1, 1)
        indices = indices.repeat(1, *broadcast_shape)
        return indices

    def compress(self, bottleneck, block_encode=True, vectorize=False):
        """
        Compresses floating-point tensors to bitsrings.

        Compresses the tensor to bit strings. `bottleneck` is first quantized
        as in `quantize()`, and then compressed using the probability tables derived
        from the entropy model. The quantized tensor can later be recovered by
        calling `decompress()`.

        Arguments:
        bottleneck: Data to be compressed. Format (N,C,H,W). An independent entropy 
                    model is trained for each channel.
           
        Returns:
        encoded:        Tensor of the same shape as `bottleneck` containing the 
                        compressed message.
        """
        input_shape = tuple(bottleneck.size())
        batch_shape = input_shape[0]
        coding_shape = input_shape[1:]
        broadcast_shape = input_shape[2:]

        indices = self.compute_indices(broadcast_shape)

        if len(indices.size()) < 4:
            indices = indices.unsqueeze(0)
            indices = torch.repeat_interleave(indices, repeats=batch_shape, dim=0)

        symbols = torch.floor(bottleneck + 0.5).to(torch.int32)
        rounded = symbols.clone()

        assert symbols.size() == indices.size(), 'Indices should have same size as inputs.'
        assert len(symbols.size()) == 4, 'Expect (N,C,H,W)-format input.'

        # All inputs should be integer-typed
        symbols = symbols.cpu().numpy()
        indices = indices.cpu().numpy()

        cdf = self.CDF.cpu().numpy().astype('uint32')
        cdf_length = self.CDF_length.cpu().numpy()
        cdf_offset = self.CDF_offset.cpu().numpy()
        
        """
        For each value in `data`, the corresponding value in `index` determines which
        probability model in `cdf` is used to encode it.

        Arguments `symbols` and `index` should have the same shape. `symbols` contains the
        values to be encoded. `index` denotes which row in `cdf` should be used to
        encode the corresponding value in `data`, and which element in `offset`
        determines the integer interval the cdf applies to. Naturally, the elements of
        `index` should be in the half-open interval `[0, cdf.shape[0])`.
        """

        encoded, coding_shape = compression_utils.ans_compress(symbols, indices, cdf, cdf_length, cdf_offset,
            coding_shape, precision=self.precision, vectorize=vectorize, 
            block_encode=block_encode)

        return encoded, coding_shape, rounded

    
    def decompress(self, encoded, batch_shape, broadcast_shape, coding_shape, vectorize=False, 
        block_decode=True):
        """
        Decompress bitstrings to floating-point tensors.

        Reconstructs the quantized tensor from bitstrings produced by `compress()`.
        It is necessary to provide a part of the output shape in `broadcast_shape`.

        Arguments:
        encoded:            Tensor containing the compressed bit strings produced by the 
                            `compress()` method.
        batch_shape         Int. Number of tensors encoded in `encoded`.
        broadcast_shape:    Iterable of ints. Spatial extent of quantized feature map.
        coding_shape:       Shape of encoded messages.

        Returns:
        decoded:            Tensor of same shape as input to `compress()`.
        """

        n_channels = self.distribution.n_channels
        # same as `input_shape` to `compress()`
        symbols_shape =  (batch_shape, n_channels, *broadcast_shape)

        indices = self.compute_indices(broadcast_shape)

        if len(indices.size()) < 4:
            indices = indices.unsqueeze(0)
            indices = torch.repeat_interleave(indices, repeats=batch_shape, dim=0)

        indices_size = tuple(indices.size())
        assert len(indices.size()) == 4, 'Expect (N,C,H,W)-format input.'
        assert batch_shape == indices.size(0), 'Batch shape mismatch!'
        assert indices_size == symbols_shape, (
            f"Index ({indices_size}) - symbol ({symbols_shape}) shape mismatch!")

        indices = indices.cpu().numpy()
        cdf = self.CDF.cpu().numpy().astype('uint32')
        cdf_length = self.CDF_length.cpu().numpy()
        cdf_offset = self.CDF_offset.cpu().numpy()

        decoded = compression_utils.ans_decompress(encoded, indices, cdf, cdf_length, cdf_offset,
            coding_shape, precision=self.precision, vectorize=vectorize, block_decode=block_decode)

        symbols = torch.Tensor(decoded)
        symbols = torch.reshape(symbols, symbols_shape)
        decoded_raw = symbols.clone()
        decoded = self.dequantize(symbols)

        return decoded, decoded_raw


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
        super(HyperpriorDensity, self).__init__(**kwargs)
        
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
        lt = compression_utils.estimate_tails(cdf_logits_func, target=-np.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()
        return lt.reshape(self.n_channels)

    def upper_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        ut = compression_utils.estimate_tails(cdf_logits_func, target=np.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()
        return ut.reshape(self.n_channels)

    def median(self):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        _median = compression_utils.estimate_tails(cdf_logits_func, target=0., 
            shape=torch.Size((self.n_channels,1,1))).detach()
        return _median.reshape(self.n_channels)

    def likelihood(self, x, collapsed_format=False, **kwargs):
        """
        Expected input: (N,C,H,W)
        """
        latents = x

        # Converts latents to (C,1,*) format

        if collapsed_format is False:
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

        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        if collapsed_format is True:
            return likelihood_

        # Reshape to (N,C,H,W)
        likelihood_ = torch.reshape(likelihood_, shape)
        likelihood_ = likelihood_.permute(1,0,2,3)

        return likelihood_

    def forward(self, x, **kwargs):
        return self.likelihood(x)


if __name__ == '__main__':

    import time

    n_channels = 64
    use_blocks = True
    vectorize = True
    hyperprior_density = HyperpriorDensity(n_channels)
    hyperprior_entropy_model = HyperpriorEntropyModel(hyperprior_density)
    hyperprior_entropy_model.build_tables()

    loc, scale = 2.401, 3.43
    n_data = 1
    toy_shape = (n_data, n_channels, 117, 185)
    bottleneck = torch.randn(toy_shape)

    bits, bpp, bpi = hyperprior_entropy_model._estimate_compression_bits(bottleneck, 
        spatial_shape=toy_shape[2:])

    start_t = time.time()
    encoded, coding_shape, symbols = hyperprior_entropy_model.compress(bottleneck,
        block_encode=use_blocks, vectorize=vectorize)

    if (use_blocks is True) or (vectorize is True): 
        enc_shape = encoded.shape[0]
    else:
        enc_shape = sum([len(enc) for enc in encoded])

    print('Encoded shape', enc_shape)

    decoded, decoded_raw = hyperprior_entropy_model.decompress(encoded, n_data,
        broadcast_shape=toy_shape[2:], coding_shape=coding_shape, block_decode=use_blocks,
        vectorize=vectorize)

    print('Decoded shape', decoded.shape)
    delta_t = time.time() - start_t
    print(f'Delta t {delta_t:.2f} s | ', torch.mean((decoded_raw == symbols).float()).item())


    cbits = enc_shape * 32
    print(f'Symbols compressed to {cbits:.1f} bits.')
    print(f'Estimated entropy {bits:.3f} bits.')

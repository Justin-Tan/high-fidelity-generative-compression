import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom
from src.compression import entropy_models

MIN_SCALE = entropy_models.MIN_SCALE
MIN_LIKELIHOOD = entropy_models.MIN_LIKELIHOOD
TAIL_MASS = entropy_models.TAIL_MASS
PRECISION_P = entropy_models.PRECISION_P


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


    def build_tables(self, offsets=None, **kwargs):

        if offsets is None:
            offsets = 0.
        
        # Shape [n_channels] or [B, n_channels]
        # Compression is typically done for individual images
        """
        More generally, dimensions which are independent but
        not necessarily identically distributed.
        """
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

        max_length = pmf_length.max()
        samples = torch.arange(max_length, dtype=self.distribution.dtype)

        # Broadcast to [n_channels,1,*] format
        samples = samples + pmf_start.view(-1,1,1)
        pmf = self.distribution.likelihood(samples, collapsed_format=True)

        # [n_channels, max_length]
        pmf = torch.reshape(pmf, (max_length, -1))
        pmf = torch.transpose(pmf, 0, 1)

        # pmf_length = tf.broadcast_to(pmf_length, self.prior_shape_tensor)
        pmf_length = torch.reshape(pmf_length, [-1])
        cdf_length = pmf_length + 2
        # cdf_offset = tf.broadcast_to(-minima, self.prior_shape_tensor)
        cdf_offset = -minima
        cdf_offset = torch.reshape(cdf_offset, [-1])
        
        # CDF shape [n_channels, max_length + 2] - account for fenceposts + overflow
        CDF = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for n, (pmf_, pmf_length_) in enumerate(zip(pmf, pmf_length)):
            pmf_ = pmf_[:pmf_length_]  # [max_length]
            overflow = torch.clamp(1. - torch.sum(pmf_, keepdim=True), min=0.)
            pmf_ = torch.cat((pmf_, overflow), dim=0)

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

        quantized = self.quantize_st(x, offsets=None)
        likelihood = self.distribution.likelihood(quantized)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def compute_indices(self, broadcast_shape):
        index_size = self.distribution.n_channels
        indices = torch.arange(index_size, dtype=torch.int32).view(-1, 1, 1)
        indices = indices.repeat(1, *broadcast_shape)
        return indices

    def compress(self, bottleneck):
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
        strings:    Tensor of the same shape as `bottleneck` containing a string for 
                    each batch entry.
        """
        input_shape = tuple(bottleneck.size())
        batch_shape = input_shape[0]
        coding_shape = input_shape[1:]
        broadcast_shape = input_shape[2:]

        indices = self.compute_indices(broadcast_shape)
        symbols = torch.round(bottleneck).to(torch.int32)

        assert symbols.size() == indices.size(), 'Indices should have same size as inputs.'
        assert len(symbols.size() == 4), 'Expect (N,C,H,W)-format input.'

        strings = torch.zeros(batch_shape)
        
        """
        For each value in `data`, the corresponding value in `index` determines which
        probability model in `cdf` is used to encode it.

        Arguments `symbols` and `index` should have the same shape. `symbols` contains the
        values to be encoded. `index` denotes which row in `cdf` should be used to
        encode the corresponding value in `data`, and which element in `offset`
        determines the integer interval the cdf applies to. Naturally, the elements of
        `index` should be in the half-open interval `[0, cdf.shape[0])`.
        """

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

    
    def decompress(self, strings, broadcast_shape):
        """
        Decompress bitstrings to floating-point tensors.

        Reconstructs the quantized tensor from bitstrings produced by `compress()`.
        It is necessary to provide a part of the output shape in `broadcast_shape`.

        Arguments:
        strings:            Tensor containing the compressed bit strings.
        broadcast_shape:    Iterable of ints. Spatial extent of quantized feature map. 

        Returns:
        reconstruction:     Tensor of same shape as input to `compress()`.
        """

        batch_shape = strings.size(0)
        n_channels = self.distribution.n_channels
        # same as `input_shape` to `compress()`
        symbols_shape =  (batch_shape, n_channels, *broadcast_shape)

        indices = self.compute_indices(broadcast_shape)
        strings = torch.reshape(strings, (-1,))

        assert len(indices.size() == 4), 'Expect (N,C,H,W)-format input.'
        assert strings.size(0) == indices.size(0), 'Batch shape mismatch!'

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
        outputs = self.dequantize(symbols)

        return outputs


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
        lt = compression_utils.estimate_tails(cdf_logits_func, target=-torch.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()

        return lt.reshape(self.n_channels)

    def upper_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        ut = compression_utils.estimate_tails(cdf_logits_func, target=torch.log(2. / tail_mass - 1.), 
            shape=torch.Size((self.n_channels,1,1))).detach()

        return ut.reshape(self.n_channels)

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

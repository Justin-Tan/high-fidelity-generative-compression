"""
Based on many sources, mainly https://github.com/j-towns/craystack/blob/master/craystack/rans.py
"""

import numpy as np

from warnings import warn

# Custom
from src.helpers import maths
from src.compression import ans as vrans
from collections import namedtuple

Codec = namedtuple('Codec', ['push', 'pop'])

OVERFLOW_WIDTH = 4

def base_codec(enc_statfun, dec_statfun, precision):

    def push(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.push(message, start, freq, precision)

    def pop(message):
        cf, pop_fun = vrans.pop(message, precision)
        symbol = dec_statfun(cf)
        start, freq = enc_statfun(symbol)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        return pop_fun(start, freq), symbol

    return Codec(push, pop)

def _indexed_cdf_to_enc_statfun(cdf_i):
    # enc_statfun: symbol |-> start, freq
    def _enc_statfun(value):
        # Value in [0, max_length]
        lower = cdf_i[value]
        # cum_freq, pmf @ value
        return lower, cdf_i[value + 1] - lower
            
    return _enc_statfun

def _indexed_cdf_to_dec_statfun(cdf_i):
    # dec_statfun: cf |-> symbol
    def _dec_statfun(cum_freq):
        # cum_freq in [0, 2 ** precision]
        # Search such that CDF[s] <= cum_freq < CDF[s+1]
        sym = np.searchsorted(cdf_i, cum_freq, side='right')
        return sym

    return _dec_statfun

def ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    overflow_width=OVERFLOW_WIDTH):
    """
    ANS-encodes unbounded integer data using an indexed probability table.

    For each value in data, the corresponding value in index determines which probability model 
    in cdf is used to encode it. The data can be arbitrary signed integers, where the integer 
    intervals determined by offset and cdf_size are modeled using the cumulative distribution 
    functions (CDF) in `cdf`. Everything else is encoded with a variable length code.

    The argument `cdf` is a 2-D tensor and its each row contains a CDF. The argument
    `cdf_size` is a 1-D tensor, and its length should be the same as the number of
    rows of `cdf`. The values in `cdf_size` denotes the length of CDF vector in the
    corresponding row of `cdf`.

    For i = 0,1,..., let `m = cdf_size[i]`. Then for j = 0,1,...,m-1,

    ```
    cdf[..., 0] / 2^precision = Pr(X < 0) = 0
    cdf[..., 1] / 2^precision = Pr(X < 1) = Pr(X <= 0)
    cdf[..., 2] / 2^precision = Pr(X < 2) = Pr(X <= 1)
    ...
    cdf[..., m-1] / 2^precision = Pr(X < m-1) = Pr(X <= m-2).
    ```

    We require that `1 < m <= cdf.shape[1]` and that all elements of `cdf` be in the
    closed interval `[0, 2^precision]`.

    Arguments `data` and `index` should have the same shape. `data` contains the
    values to be encoded. `index` denotes which row in `cdf` should be used to
    encode the corresponding value in `data`, and which element in `offset`
    determines the integer interval the cdf applies to. Naturally, the elements of
    `index` should be in the half-open interval `[0, cdf.shape[0])`.

    When a value from `data` is in the interval `[offset[i], offset[i] + m - 2)`,
    then the value is range encoded using the CDF values. The last entry in each
    CDF (the one at `m - 1`) is an overflow code. When a value from `data` is
    outside of the given interval, the overflow value is encoded, followed by a
    variable-length encoding of the actual data value.

    The encoded output contains neither the shape information of the encoded data
    nor a termination symbol. Therefore the shape of the encoded data must be
    explicitly provided to the decoder.

    symbols <-> indices
    cdf <-> cdf_offset <-> cdf_length
    """

    if np.any(precision >= 24):
        warn('Detected precision over 24. Codecs lose accuracy at high '
             'precision.')

    message = vrans.empty_message(())
    symbols = symbols.astype(np.int32).flatten()
    indices = indices.astype(np.int32).flatten()

    for i in range(len(symbols)):  # loop over batch dimension

        cdf_index = indices[i]

        assert (cdf_index >= 0 and cdf_index < cdf.shape[0]), (
            f"Invalid index {cdf_index} for symbol {i}")

        max_value = cdf_length[cdf_index] - 2

        assert max_value >= 0 and max_value < cdf.shape[1] - 1, (
            f"Invalid max length {max_value} for symbol {i}")

        # Map values with tracked probabilities to range [0, ..., max_value]
        value = symbols[i]
        value -= cdf_offset[cdf_index]

        """
        Handle overflows here
        """

        # If outside of this range, map value to non-negative integer overflow.
        if (value < 0):
            overflow = -2 * value - 1;
        elif (value >= max_value):
            overflow = 2 * (value - max_value);
        value = max_value

        assert value >= 0 and value < cdf_length[cdf_index] - 1, (
            f"Invalid shifted value {value} for symbol {i} w/ "
            f"cdf_length {cdf_length[cdf_index]}")


        # Bin of discrete CDF that value belongs to
        cdf_i = cdf[cdf_index]

        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _indexed_cdf_to_dec_statfun(cdf_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        message = symbol_push(message, value)

        """
        Also encode overflows here
        """

    encoded = vrans.flatten(message)
    message_length = len(encoded)
    print('Symbol compressed to {:.3f} bits.'.format(32 * message_length))

    return encoded

def vec_ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, coding_shape,
    overflow_width=OVERFLOW_WIDTH):
    """
    Vectorized version. Incurs constant bit overhead, but is faster.

    ANS-encodes unbounded integer data using an indexed probability table.
    """

    if np.any(precision >= 24):
        warn('Detected precision over 24. Codecs lose accuracy at high '
             'precision.')

    B = symbols.size(0)
    symbols = symbols.to(np.int32)
    indices = indices.to(np.int32)

    cdf_index = indices
        
    assert bool(np.all(cdf_index >= 0)) and bool(np.all(cdf_index < cdf.shape[0])), (
        "Invalid index.")

    max_value = cdf_length[cdf_index] - 2

    assert bool(np.all(max_value >= 0)) and bool(np.all(max_value < cdf.shape[1] - 1)), (
        "Invalid max length.")

    # Map values with tracked probabilities to range [0, ..., max_value]
    value = symbols - cdf_offset[cdf_index]

    assert bool(np.all(value >= 0)) and bool(np.all(value < cdf_length[i] - 1)), (
        "Invalid shifted value for current symbol.")

    """
    Handle overflows here
    """

    message = vrans.empty_message(coding_shape)

    for i in range(B):

        # Bin of discrete CDF that value belongs to
        cdf_i = cdf[cdf_index]

        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _indexed_cdf_to_dec_statfun(cdf_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        message = symbol_push(message, value)


        """
        Also encode overflows here
        """

    encoded = vrans.flatten(message)
    message_length = len(encoded)

    print('{} symbols compressed to {:.3f} bits.'.format(B, 32 * message_length))

    return encoded


def ans_index_decoder(encoded, indices, cdf, cdf_length, cdf_offset, precision):

    """
    Reverse op of `ans_index_encoder`. Decodes ans-encoded bitstring into a decoded message tensor.
    Arguments (`indices`, `cdf`, `cdf_length`, `cdf_offset`, `precision`) must be identical to the
    inputs to `ans_index_encoder` used to generate the encoded tensor.
    """

    if np.any(precision >= 24):
        warn('Detected precision over 24. Codecs lose accuracy at high '
             'precision.')

def ans_encode_decode_test():

    np.testing.assert_almost_equal(symbols, decompressed_symbols)
    np.testing.assert_almost_equal(message, decompressed_message)


if __name__ == '__main__':

    import torch
    import torch.nn.functional as F
    import random

    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    def prior_scale_table(scales_min=SCALES_MIN, scales_max=SCALES_MAX, levels=SCALES_LEVELS):
        scale_table = np.exp(np.linspace(np.log(scales_min), np.log(scales_max), levels))
        return scale_table

    def compute_indices(scales, scale_table):
        # Compute the indexes into the table for each of the passed-in scales.
        indices = torch.ones_like(scales, dtype=torch.int32) * (len(scale_table) - 1)

        for s in scale_table[:-1]:
            indices = indices - (scales <= s).to(torch.int32) 

        return indices   

    _standardized_quantile = maths.standardized_quantile_gaussian
    _standardized_cumulative = maths.standardized_CDF_gaussian

    def build_tables(scale_table, tail_mass=2**(-8), precision=16):
        
        scale_table = torch.Tensor(scale_table)
        multiplier = -_standardized_quantile(tail_mass / 2)
        pmf_center = torch.ceil(scale_table * multiplier).to(torch.int32)
        pmf_length = 2 * pmf_center + 1
        # [n_scales]
        max_length = torch.max(pmf_length).item()

        samples = torch.abs(torch.arange(max_length).int() - pmf_center[:, None]).float()
        samples_scale = scale_table.unsqueeze(1).float()

        upper = _standardized_cumulative((.5 - samples) / samples_scale)
        lower = _standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower  # [n_scales, max_length]

        # [n_scales]
        tail_mass = 2 * lower[:, :1]
        cdf_length = pmf_length + 2
        cdf_offset = -pmf_center  # How far away we have to go to pass the tail mass
        cdf_length = cdf_length.to(torch.int32)
        cdf_offset = cdf_offset.to(torch.int32)

        # CDF shape [n_scales,  max_length + 2] - account for fenceposts + overflow
        CDF = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for n, (pmf_, pmf_length_, tail_) in enumerate(zip(pmf, pmf_length, tail_mass)):
            pmf_ = pmf_[:pmf_length_]  # [max_length]
            overflow = torch.clamp(1. - torch.sum(pmf_, dim=0, keepdim=True), min=0.)  
            # pmf_ = torch.cat((pmf_, overflow), dim=0)
            pmf_ = torch.cat((pmf_, tail_), dim=0)

            cdf_ = maths.pmf_to_quantized_cdf(pmf_, precision)
            cdf_ = F.pad(cdf_, (0, max_length - pmf_length_), mode='constant', value=0)
            CDF[n] = cdf_

        return CDF, cdf_length, cdf_offset

    toy_shape = (5,42,7,7)
    bottleneck = torch.randn(toy_shape)
    scales = torch.randn(toy_shape)*np.sqrt(4.2) + 4.2
    scales = torch.clamp(scales, min=SCALES_MIN)

    scale_table = prior_scale_table()
    indices = compute_indices(scales, scale_table)

    cdf, cdf_length, cdf_offset = build_tables(scale_table)

    input_shape = tuple(bottleneck.size())
    batch_shape = input_shape[0]
    coding_shape = input_shape[1:]  # (C,H,W)
    symbols = torch.round(bottleneck).to(torch.int32)

    ans_index_encoder(symbols.numpy(), indices.numpy(), cdf.numpy().astype('uint64'), 
        cdf_length.numpy(), cdf_offset.numpy(), precision=16)
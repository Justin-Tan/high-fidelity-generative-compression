"""
Based on many sources, mainly:

Fabian Gielsen's ANS implementation: https://github.com/rygorous/ryg_rans
Craystack ANS implementation: https://github.com/j-towns/craystack/blob/master/craystack/rans.py
"""

OVERFLOW_WIDTH = 4
OVERFLOW_CODE = 1 << (1 << OVERFLOW_WIDTH)
PATCH_SIZE = (1,1)

import torch
import numpy as np

from warnings import warn
from collections import namedtuple

# Custom
from src.helpers import maths, utils
from src.compression import ans as vrans
from src.compression import compression_utils

Codec = namedtuple('Codec', ['push', 'pop'])
cast2u64 = lambda x: np.array(x, dtype=np.uint64)

def base_codec(enc_statfun, dec_statfun, precision, log=False):
    if np.any(precision >= 24):
        warn('Detected precision over 28. Codecs lose accuracy at high '
             'precision.')

    def push(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.push(message, start, freq, precision)

    def pop(message, log=log):
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
        return lower, cdf_i[int(value + np.uint64(1))] - lower
            
    return _enc_statfun

def _vec_indexed_cdf_to_enc_statfun(cdf_i):
    # enc_statfun: symbol |-> start, freq
    def _enc_statfun(value):
        # (coding_shape) = (C,H,W) by default but canbe generalized
        # cdf_i: [(coding_shape), pmf_length + 2]
        # value: [(coding_shape)]
        lower = np.take_along_axis(cdf_i, 
            np.expand_dims(value, -1), axis=-1)[..., 0]
        upper = np.take_along_axis(cdf_i, 
            np.expand_dims(value + 1, -1), axis=-1)[..., 0]
        return lower, upper - lower

    return _enc_statfun

def _indexed_cdf_to_dec_statfun(cdf_i, cdf_i_length):
    # dec_statfun: cf |-> symbol
    cdf_i = cdf_i[:cdf_i_length]
    term = cdf_i[-1]
    assert term == OVERFLOW_CODE or term == 1 << OVERFLOW_WIDTH, (
        f"{cdf_i[-1]} expected to be overflow value."
    )

    def _dec_statfun(cum_freq):
        # Search such that CDF[s] <= cum_freq < CDF[s+1]
        sym = np.searchsorted(cdf_i, cum_freq, side='right') - 1
        return sym

    return _dec_statfun

def _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_i_length):
    # dec_statfun: cf |-> symbol
    *coding_shape, max_cdf_length = cdf_i.shape
    coding_shape = tuple(coding_shape)
    cdf_i_flat = np.reshape(cdf_i, (-1, max_cdf_length))

    cdf_i_flat_ragged = [c[:l] for (c,l) in zip(cdf_i_flat, 
        cdf_i_length.flatten())]

    def _dec_statfun(value):
        # (coding_shape) = (C,H,W) by default but can be generalized
        # cdf_i: [(coding_shape), pmf_length + 2]
        # value: [(coding_shape)]
        assert value.shape == coding_shape, (
            f"CDF-value shape mismatch! {value.shape} v. {coding_shape}")
        sym_flat = np.array(
            [np.searchsorted(cb, v_i, 'right') - 1 for (cb, v_i) in 
                zip(cdf_i_flat_ragged, value.flatten())])

        sym = np.reshape(sym_flat, coding_shape)
        return sym  # (coding_shape)

    return _dec_statfun

def ans_index_buffered_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    overflow_width=OVERFLOW_WIDTH, **kwargs):

    """
    Based on "https://github.com/tensorflow/compression/blob/master/tensorflow_compression/cc/
    kernels/unbounded_index_range_coding_kernels.cc"
    
    ANS-encodes unbounded integer data using an indexed probability table. 
    Pushes instructions for encoding scalars sequentially to a buffer.

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

    instructions = []

    coding_shape = symbols.shape[1:]
    symbols = symbols.astype(np.int32).flatten()
    indices = indices.astype(np.int32).flatten()

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)

    enc_statfun_overflow = _indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _indexed_cdf_to_dec_statfun(overflow_cdf,
        len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow,
        dec_statfun_overflow, overflow_width)

    # LIFO - last item in buffer is first item decompressed
    for i in range(len(indices)):  # loop over flattened axis

        cdf_index = indices[i]
        cdf_i = cdf[cdf_index]
        cdf_length_i = cdf_length[cdf_index]

        assert (cdf_index >= 0 and cdf_index < cdf.shape[0]), (
            f"Invalid index {cdf_index} for symbol {i}")

        max_value = cdf_length_i - 2

        assert max_value >= 0 and max_value < cdf.shape[1] - 1, (
            f"Invalid max length {max_value} for symbol {i}")

        # Data in range [offset[cdf_index], offset[cdf_index] + m - 2] is ANS-encoded
        # Map values with tracked probabilities to range [0, ..., max_value]
        value = symbols[i]
        value -= cdf_offset[cdf_index]

        # If outside of this range, map value to non-negative integer overflow.
        overflow = 0
        if (value < 0):
            overflow = -2 * value - 1
            value = max_value
        elif (value >= max_value):
            overflow = 2 * (value - max_value)
            value = max_value

        assert value >= 0 and value < cdf_length_i - 1, (
            f"Invalid shifted value {value} for symbol {i} w/ "
            f"cdf_length {cdf_length[cdf_index]}")

        # Bin of discrete CDF that value belongs to
        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        start, freq = enc_statfun(value)
        instructions.append((start, freq, False))

        # When value is outside of the given interval, the overflow value is encoded,
        # followed by a variable-length encoding of the actual data value.
        if value == max_value:
            # pass
            widths = 0
            while ((overflow >> (widths * overflow_width)) != 0):
                widths += 1

            val = widths
            while (val >= max_overflow):
                start, freq = enc_statfun_overflow(cast2u64(max_overflow))
                instructions.append((start, freq, True))
                val -= max_overflow
            
            start, freq = enc_statfun_overflow(cast2u64(val))
            instructions.append((start, freq, True))

            for j in range(widths):
                val = (overflow >> (j * overflow_width)) & max_overflow
                start, freq = enc_statfun_overflow(cast2u64(val))
                instructions.append((start, freq, True))

    return instructions, coding_shape

def ans_index_encoder_flush(instructions, precision, overflow_width=OVERFLOW_WIDTH, **kwargs):

    message = vrans.empty_message(())

    # LIFO - last item compressed is first item decompressed
    for i in reversed(range(len(instructions))):
        
        start, freq, flag = instructions[i]

        if flag is False:
            message = vrans.push(message, start, freq, precision)
        else:
            message = vrans.push(message, start, freq, overflow_width)

    encoded = vrans.flatten(message)
    message_length = len(encoded)
    print('Symbol compressed to {:.3f} bits.'.format(32 * message_length))
    return encoded

def ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    overflow_width=OVERFLOW_WIDTH, **kwargs):

    instructions, coding_shape = ans_index_buffered_encoder(symbols, indices, cdf,
        cdf_length, cdf_offset, precision, overflow_width)

    encoded = ans_index_encoder_flush(instructions, precision, overflow_width)

    return encoded, coding_shape


def vec_ans_index_buffered_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):
    """
    Vectorized version of `ans_index_encoder`. Incurs constant bit overhead, 
    but is faster.

    ANS-encodes unbounded integer data using an indexed probability table.
    """
    
    instructions = []

    symbols_shape = symbols.shape
    B, n_channels = symbols_shape[:2]
    symbols = symbols.astype(np.int32)
    indices = indices.astype(np.int32)
    cdf_index = indices

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)[None, None, None, :]

    enc_statfun_overflow = _vec_indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _vec_indexed_cdf_to_dec_statfun(overflow_cdf,
        np.ones_like(overflow_cdf) * len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow,
        dec_statfun_overflow, overflow_width)

    assert bool(np.all(cdf_index >= 0)) and bool(np.all(cdf_index < cdf.shape[0])), (
        "Invalid index.")

    max_value = cdf_length[cdf_index] - 2

    assert bool(np.all(max_value >= 0)) and bool(np.all(max_value < cdf.shape[1] - 1)), (
        "Invalid max length.")

    # Map values with tracked probabilities to range [0, ..., max_value]
    values = symbols - cdf_offset[cdf_index]

    # If outside of this range, map value to non-negative integer overflow.
    overflow = np.zeros_like(values)
    of_mask_lower = values < 0
    overflow = np.where(of_mask_lower, -2 * values - 1, overflow)
    of_mask_upper = values >= max_value
    overflow = np.where(of_mask_upper, 2 * (values - max_value), overflow)
    values = np.where(np.logical_or(of_mask_lower, of_mask_upper), max_value, values)

    assert bool(np.all(values >= 0)), (
        "Invalid shifted value for current symbol - values must be non-negative.")

    assert bool(np.all(values < cdf_length[cdf_index] - 1)), (
        "Invalid shifted value for current symbol - outside cdf index bounds.")

    if B == 1:
        # Vectorize on patches - there's probably a way to interlace patches with
        # batch elements for B > 1 ...
        if ((symbols_shape[2] % PATCH_SIZE[0] == 0) and (symbols_shape[3] % PATCH_SIZE[1] == 0)) is False:
            values = utils.pad_factor(torch.Tensor(values), symbols_shape[2:], 
                factor=PATCH_SIZE).cpu().numpy().astype(np.int32)
            indices = utils.pad_factor(torch.Tensor(indices), symbols_shape[2:], 
                factor=PATCH_SIZE).cpu().numpy().astype(np.int32)
            overflow = utils.pad_factor(torch.Tensor(overflow), symbols_shape[2:], 
                factor=PATCH_SIZE).cpu().numpy().astype(np.int32)

        assert (values.shape[2] % PATCH_SIZE[0] == 0) and (values.shape[3] % PATCH_SIZE[1] == 0)
        assert (indices.shape[2] % PATCH_SIZE[0] == 0) and (indices.shape[3] % PATCH_SIZE[1] == 0)
  
        values, _ = compression_utils.decompose(values, n_channels)
        overflow, _ = compression_utils.decompose(overflow, n_channels)
        cdf_index, unfolded_shape = compression_utils.decompose(indices, n_channels)
        coding_shape = values.shape[1:]
        assert coding_shape == cdf_index.shape[1:]


    # LIFO - last item in buffer is first item decompressed
    for i in range(len(cdf_index)):  # loop over batch dimension
        # Bin of discrete CDF that value belongs to
        value_i = values[i]
        cdf_index_i = cdf_index[i]        
        cdf_i = cdf[cdf_index_i]
        cdf_length_i = cdf_length[cdf_index_i]
        max_value_i = cdf_length_i - 2

        enc_statfun = _vec_indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        start, freq = enc_statfun(value_i)
        instructions.append((start, freq, False, precision, 0))

        """
        Encode overflows here
        """
        # No-op
        empty_start = np.zeros_like(value_i).astype(np.uint)
        empty_freq = np.ones_like(value_i).astype(np.uint)

        overflow_i = overflow[i]
        of_mask = value_i == max_value_i

        if np.any(of_mask):

            widths = np.zeros_like(value_i)
            cond_mask = (overflow_i >> (widths * overflow_width)) != 0

            while np.any(cond_mask):
                widths = np.where(cond_mask, widths+1, widths)
                cond_mask = (overflow_i >> (widths * overflow_width)) != 0

            val = widths
            cond_mask = val >= max_overflow
            while np.any(cond_mask):
                print('Warning: Undefined behaviour.')
                val_push = cast2u64(max_overflow)
                overflow_start, overflow_freq = enc_statfun_overflow(val_push)
                start = overflow_start[of_mask]
                freq = overflow_start[of_mask]
                instructions.append((start, freq, True, int(overflow_width), of_mask))
                # val[cond_mask] -= max_overflow
                val = np.where(cond_mask, val-max_overflow, val)
                cond_mask = val >= max_overflow

            val_push = cast2u64(val)
            overflow_start, overflow_freq = enc_statfun_overflow(val_push)
            start = overflow_start[of_mask]
            freq = overflow_freq[of_mask]
            instructions.append((start, freq, True, int(overflow_width), of_mask))

            cond_mask = widths != 0
            while np.any(cond_mask):
                counter = 0
                encoding = (overflow_i >> (counter * overflow_width)) & max_overflow
                val = np.where(cond_mask, encoding, val)
                val_push = cast2u64(val)
                overflow_start, overflow_freq = enc_statfun_overflow(val_push)
                start = overflow_start[of_mask]
                freq = overflow_freq[of_mask]
                instructions.append((start, freq, True, int(overflow_width), of_mask))
                widths = np.where(cond_mask, widths-1, widths)
                cond_mask = widths != 0
                counter += 1

    return instructions, coding_shape


def overflow_view(value, mask):
    return value[mask]

def substack(codec, view_fun):
    """
    Apply a codec on a subset of a message head.
    view_fun should be a function: head -> subhead, for example
    view_fun = lambda head: head[0]
    to run the codec on only the first element of the head
    """
    def push(message, start, freq, precision, mask):
        head, tail = message
        view_fun_ = lambda x: view_fun(x, mask)
        subhead, update = compression_utils.view_update(head, view_fun_)
        subhead, tail = vrans.push((subhead, tail), start, freq, precision)
        return update(subhead), tail

    def pop(message, precision, mask, *args, **kwargs):
        head, tail = message
        view_fun_ = lambda x: view_fun(x, mask)
        subhead, update = compression_utils.view_update(head, view_fun_)

        cf, pop_fun = vrans.pop((subhead, tail), precision)
        symbol = cf
        start, freq = symbol, 1
        
        assert np.all(start <= cf) and np.all(cf < start + freq)
        (subhead, tail), data = pop_fun(start, freq), symbol
        updated_head = update(subhead)
        return (updated_head, tail), data

    return Codec(push, pop)

def vec_ans_index_encoder_flush(instructions, precision, coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):

    message = vrans.empty_message(coding_shape)
    overflow_push, _ = substack(codec=None, view_fun=overflow_view)
    # LIFO - last item compressed is first item decompressed
    for i in reversed(range(len(instructions))):
        
        start, freq, flag, precision_i, mask = instructions[i]

        if flag is False:
            message = vrans.push(message, start, freq, precision)
        else:
            # Substack on overflow values
            overflow_precision = precision_i
            message = overflow_push(message, start, freq, overflow_precision, mask)

    encoded = vrans.flatten(message)
    message_length = len(encoded)
    print('Symbol compressed to {:.3f} bits.'.format(32 * message_length))
    return encoded

def vec_ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):

    instructions, coding_shape = vec_ans_index_buffered_encoder(symbols, indices, cdf,
        cdf_length, cdf_offset, precision, coding_shape, overflow_width)

    encoded = vec_ans_index_encoder_flush(instructions, precision, coding_shape, overflow_width)

    return encoded, coding_shape

def ans_index_decoder(encoded, indices, cdf, cdf_length, cdf_offset, precision,
    coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):

    """
    Reverse op of `ans_index_encoder`. Decodes ans-encoded bitstring `encoded` into 
    a decoded message tensor `decoded.

    Arguments (`indices`, `cdf`, `cdf_length`, `cdf_offset`, `precision`) must be 
    identical to the inputs to the encoding function used to generate the encoded 
    tensor.
    """

    message = vrans.unflatten_scalar(encoded)  # (head, tail)
    decoded = np.empty(indices.shape).flatten()
    indices = indices.astype(np.int32).flatten()

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)

    enc_statfun_overflow = _indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _indexed_cdf_to_dec_statfun(overflow_cdf,
        len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow,
        dec_statfun_overflow, overflow_width)

    for i in range(len(indices)):

        cdf_index = indices[i]
        cdf_i = cdf[cdf_index]
        cdf_length_i = cdf_length[cdf_index]

        assert (cdf_index >= 0 and cdf_index < cdf.shape[0]), (
            f"Invalid index {cdf_index} for symbol {i}")

        max_value = cdf_length_i - 2

        assert max_value >= 0 and max_value < cdf.shape[1] - 1, (
            f"Invalid max length {max_value} for symbol {i}")

        # Bin of discrete CDF that value belongs to
        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        message, value = symbol_pop(message)

        """
        Handle overflow values
        """

        if value == max_value:
            # pass
            message, val = overflow_pop(message)
            val = int(val)
            widths = val

            while val == max_overflow:
                message, val = overflow_pop(message)
                val = int(val)
                widths += val
            
            overflow = 0
            for j in range(widths):
                message, val = overflow_pop(message)
                val = int(val)
                assert val <= max_overflow
                overflow |= val << (j * overflow_width)

            # Map positive values back to integer values.
            value = overflow >> 1
            if (overflow & 1):
                value = -value - 1
            else:
                value += max_value
        
        symbol = value + cdf_offset[cdf_index]
        decoded[i] = symbol

    return decoded


def vec_ans_index_decoder(encoded, indices, cdf, cdf_length, cdf_offset, precision, 
    coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):

    """
    Reverse op of `vec_ans_index_encoder`. Decodes ans-encoded bitstring into a decoded 
    message tensor.
    Arguments (`indices`, `cdf`, `cdf_length`, `cdf_offset`, `precision`) must be 
    identical to the inputs to `vec_ans_index_encoder` used to generate the encoded tensor.
    """

    original_shape = indices.shape
    B, n_channels, *_ = original_shape
    message = vrans.unflatten(encoded, coding_shape)
    indices = indices.astype(np.int32)
    cdf_index = indices

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)[None, :]

    enc_statfun_overflow = _vec_indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _vec_indexed_cdf_to_dec_statfun(overflow_cdf,
        np.ones_like(overflow_cdf) * len(overflow_cdf))
    overflow_codec = base_codec(enc_statfun_overflow,
        dec_statfun_overflow, overflow_width)

    assert bool(np.all(cdf_index >= 0)) and bool(np.all(cdf_index < cdf.shape[0])), (
        "Invalid index.")

    max_value = cdf_length[cdf_index] - 2

    assert bool(np.all(max_value >= 0)) and bool(np.all(max_value < cdf.shape[1] - 1)), (
        "Invalid max length.")

    if B == 1:
        # Vectorize on patches - there's probably a way to interlace patches with
        # batch elements for B > 1 ...

        if ((original_shape[2] % PATCH_SIZE[0] == 0) and (original_shape[3] % PATCH_SIZE[1] == 0)) is False:
            indices = utils.pad_factor(torch.Tensor(indices), original_shape[2:], 
                factor=PATCH_SIZE).cpu().numpy().astype(np.int32)
        padded_shape = indices.shape
        assert (indices.shape[2] % PATCH_SIZE[0] == 0) and (indices.shape[3] % PATCH_SIZE[1] == 0)
        cdf_index, unfolded_shape = compression_utils.decompose(indices, n_channels)
        coding_shape = cdf_index.shape[1:]


    symbols = []
    _, overflow_pop = substack(codec=overflow_codec, view_fun=overflow_view)

    for i in range(len(cdf_index)):
        cdf_index_i = cdf_index[i]
        cdf_i = cdf[cdf_index_i]
        cdf_length_i = cdf_length[cdf_index_i]

        enc_statfun = _vec_indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        message, value = symbol_pop(message)

        max_value_i = cdf_length_i - 2
        of_mask = value == max_value_i

        if np.any(of_mask):
            
            message, val = overflow_pop(message, overflow_width, of_mask)
            val = cast2u64(val)
            widths = val

            cond_mask = val == max_overflow
            while np.any(cond_mask):
                message, val = overflow_pop(message, overflow_width, of_mask)
                val = cast2u64(val)
                widths = np.where(cond_mask, widths + val, widths)
                cond_mask = val == max_overflow

            overflow = np.zeros_like(val)
            cond_mask = widths != 0
            
            while np.any(cond_mask):
                counter = 0
                message, val = overflow_pop(message, overflow_width, of_mask)
                val = cast2u64(val)
                assert np.all(val <= max_overflow)

                op = overflow | (val << (counter * overflow_width))
                overflow = np.where(cond_mask, op, overflow)
                widths = np.where(cond_mask, widths-1, widths)
                cond_mask = widths != 0
                counter += 1

            overflow_broadcast = value
            overflow_broadcast[of_mask] = overflow
            overflow = overflow_broadcast
            value = np.where(of_mask, overflow >> 1, value)
            cond_mask = np.logical_and(of_mask, overflow & 1)
            value = np.where(cond_mask, -value - 1, value)
            cond_mask = np.logical_and(of_mask, np.logical_not(overflow & 1))
            value = np.where(cond_mask, value + max_value_i, value)

        symbol = value + cdf_offset[cdf_index_i]
        symbols.append(symbol)

        
    if B == 1:
        decoded = compression_utils.reconstitute(np.stack(symbols, axis=0), padded_shape, unfolded_shape)

        if tuple(decoded.shape) != tuple(original_shape):
            decoded = decoded[:, :, :original_shape[2], :original_shape[3]]
    else:
        decoded = np.stack(symbols, axis=0)
    return decoded

def ans_encode_decode_test(symbols, decompressed_symbols):
    return np.testing.assert_almost_equal(symbols, decompressed_symbols)

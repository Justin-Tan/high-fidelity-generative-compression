def ans_index_encoder_reversed(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    overflow_width=OVERFLOW_WIDTH, **kwargs):

    message = vrans.empty_message(())
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

    # LIFO - last item compressed is first item decompressed
    for i in reversed(range(len(indices))):  # loop over flattened axis

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

        message = symbol_push(message, value)

        if value == max_value:
            pass

    encoded = vrans.flatten(message)
    message_length = len(encoded)
    return encoded, coding_shape

def vec_ans_index_encoder_reversed(symbols, indices, cdf, cdf_length, cdf_offset, precision, 
    coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):
    """
    Vectorized version of `ans_index_encoder`. Incurs constant bit overhead, 
    but is faster.

    ANS-encodes unbounded integer data using an indexed probability table.
    """
    
    symbols_shape = symbols.shape
    B, n_channels = symbols_shape[:2]
    symbols = symbols.astype(np.int32)
    indices = indices.astype(np.int32)
    cdf_index = indices
        
    assert bool(np.all(cdf_index >= 0)) and bool(np.all(cdf_index < cdf.shape[0])), (
        "Invalid index.")

    max_value = cdf_length[cdf_index] - 2

    assert bool(np.all(max_value >= 0)) and bool(np.all(max_value < cdf.shape[1] - 1)), (
        "Invalid max length.")

    # Map values with tracked probabilities to range [0, ..., max_value]
    values = symbols - cdf_offset[cdf_index]

    # If outside of this range, map value to non-negative integer overflow.
    overflow = np.zeros_like(values)

    of_mask = values < 0
    overflow = np.where(of_mask, -2 * values - 1, overflow)
    values = np.where(of_mask, max_value, values)

    of_mask = values >= max_value
    overflow = np.where(of_mask, 2 * (values - max_value), overflow)
    values = np.where(of_mask, max_value, values)

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

        assert (values.shape[2] % PATCH_SIZE[0] == 0) and (values.shape[3] % PATCH_SIZE[1] == 0)
        assert (indices.shape[2] % PATCH_SIZE[0] == 0) and (indices.shape[3] % PATCH_SIZE[1] == 0)
  
        values, _ = compression_utils.decompose(values, n_channels)
        cdf_index, unfolded_shape = compression_utils.decompose(indices, n_channels)
        coding_shape = values.shape[1:]

    message = vrans.empty_message(coding_shape)

    # LIFO - last item compressed is first item decompressed
    for i in reversed(range(len(cdf_index))):  # loop over batch dimension
        # Bin of discrete CDF that value belongs to
        value_i = values[i]
        cdf_index_i = cdf_index[i]        
        cdf_i = cdf[cdf_index_i]
        cdf_i_length = cdf_length[cdf_index_i]

        enc_statfun = _vec_indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_i_length)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)

        message = symbol_push(message, value_i)

        """
        Encode overflows here
        """

    encoded = vrans.flatten(message)
    message_length = len(encoded)

    # print('{} symbols compressed to {:.3f} bits.'.format(B, 32 * message_length))

    return encoded, coding_shape

"""
64-bit rANS encoder/decoder
Based on https://arxiv.org/abs/1402.3392

x: compressed message, represented by current state of the encoder/decoder.

precision: the natural numbers are divided into ranges of size 2^precision.
start & freq: start indicates the beginning of the range in [0, 2^precision-1]
that the current symbol is represented by. freq is the length of the range for
the given symbol.

The probability distribution is quantized to 2^precision, where 
P(symbol) ~= freq(symbol) / 2^precision

Compressed state is represented as a stack (head, tail)
"""

import numpy as np
import torch

RANS_L = 1 << 31  # the lower bound of the normalisation interval

def empty_message(shape):
    return (np.full(shape, RANS_L, "uint64"), ())

def stack_extend(stack, arr):
    return arr, stack

def stack_slice(stack, n):
    # Pop elements from message stack if
    # decoded value outside normalisation
    # interval
    slc = []
    while n > 0:
        arr, stack = stack
        if n >= len(arr):
            slc.append(arr)
            n -= len(arr)
        else:
            slc.append(arr[:n])
            stack = arr[n:], stack
            break
    return stack, np.concatenate(slc)

def push(x, starts, freqs, precisions):
    """
    Encode a vector of symbols in x. Each symbol has range given by
    [start, start + freq). All frequencies are assumed to sum to 
    "1 << precision", and the resulting bits get written to x.

    Inputs:
        x:          Compressed message of form (head, tail)  
        starts:     Starts of interval corresponding to symbols. Analogous to
                    CDF evaluated at each symbol.
        freqs:      Width of intervals corresponding to symbols.
        precision:  Determines normalization factor of probability distribution.
    """
    head, tail = x
    assert head.shape == starts.shape == freqs.shape, (
        f"Inconsistent encoder shapes! head: {head.shape} | "
        f"starts: {starts.shape} | freqs: {freqs.shape}")
    
    # 32-bit Renormalization - restrict symbols to pre-images
    x_max = ((RANS_L >> precisions) << 32) * freqs
    idxs = head >= x_max

    if np.any(idxs) > 0:
        # Push lower order bits of message onto message stack
        tail = stack_extend(tail, np.uint32(head[idxs]))  # Can also modulo with bitand
        head = np.copy(head)  # Ensure no side-effects
        head[idxs] >>= 32
    head_div_freqs, head_mod_freqs = np.divmod(head, freqs)
    return (head_div_freqs << np.uint(precisions)) + head_mod_freqs + starts, tail

def pop(x, precisions):
    head_, tail_ = x
    head_ = np.uint64(head_)
    # Modulo as bitwise and
    interval_starts = head_ & np.uint((1 << precisions) - 1)
    def pop(starts, freqs):
        head = freqs * (head_ >> np.uint(precisions)) + interval_starts - starts
        idxs = head < RANS_L
        n = np.sum(idxs)
        if n > 0:
            tail, new_head = stack_slice(tail_, n)
            # Move popped integers into lower-order
            # bits of head
            try:
                head[idxs] = (head[idxs] << np.uint(32)) | new_head
            except TypeError:
                head = (head << np.uint(32)) | new_head
        else:
            tail = tail_
        return head, tail
    
    return interval_starts, pop

def flatten(x):
    """Flatten a vrans state x into a 1d numpy array."""
    head, x = np.ravel(x[0]), x[1]
    out = [np.uint32(head >> 32), np.uint32(head)]
    while x:
        head, x = x
        out.append(head)
    return np.concatenate(out)

def unflatten(arr, shape):
    """Unflatten a 1d numpy array into a vrans state."""
    size = np.prod(shape)
    head = np.uint64(arr[:size]) << 32 | np.uint64(arr[size:2 * size])
    return np.reshape(head, shape), (arr[2 * size:], ())

def unflatten_scalar(arr):
    """Unflatten a 1d numpy array into a vrans state."""
    head = np.uint64(arr[0]) << np.uint64(32) | np.uint64(arr[1])
    return head, (arr[2:], ())

def message_equal(message1, message2):
    return np.all(flatten(message1) == flatten(message2))
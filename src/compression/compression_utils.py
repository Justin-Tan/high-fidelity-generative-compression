import torch
import math
import numpy as np
import functools
import os
import autograd.numpy as np

from autograd import make_vjp
from autograd.extend import vspace, VSpace
from collections import namedtuple

from src.helpers import utils
from src.compression import entropy_coding

# Random bits for fencing in bitstream
_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'

PATCH_SIZE = entropy_coding.PATCH_SIZE

CompressionOutput = namedtuple("CompressionOutput",
   ["hyperlatents_encoded",
    "latents_encoded",
    "hyperlatent_spatial_shape",
    "batch_shape",
    "spatial_shape",
    "hyper_coding_shape",
    "latent_coding_shape"]
)

def estimate_tails(cdf, target, shape, dtype=torch.float32, extra_counts=24):
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
    shape: The shape of the tensor representing `x`.
    Returns:
    A `torch.Tensor` representing the solution (`x`).
    """
    # A bit hacky
    lr, eps = 1e-2, 1e-8
    beta_1, beta_2 = 0.9, 0.99

    # Tails should be monotonically increasing
    device = utils.get_device()
    tails = torch.zeros(shape, dtype=dtype, requires_grad=True, device=device)

    m = torch.zeros(shape, dtype=dtype)
    v = torch.ones(shape, dtype=dtype)
    counts = torch.zeros(shape, dtype=torch.int32)

    while torch.min(counts) < extra_counts:
        loss = abs(cdf(tails) - target)
        loss.backward(torch.ones_like(tails))

        tgrad = tails.grad.cpu()

        with torch.no_grad():
            m = beta_1 * m + (1. - beta_1) * tgrad
            v = beta_2 * v + (1. - beta_2) * torch.square(tgrad)
            tails -= (lr * m / (torch.sqrt(v) + eps)).to(device)

        # Condition assumes tails init'd at zero
        counts = torch.where(torch.logical_or(counts > 0, tgrad.cpu() * tails.cpu() > 0), 
            counts+1, counts)

        tails.grad.zero_()

    return tails

def view_update(data, view_fun):
    view_vjp, item = make_vjp(view_fun)(data)
    item_vs = vspace(item)
    def update(new_item):
        assert item_vs == vspace(new_item), \
            "Please ensure new_item shape and dtype match the data view."
        diff = view_vjp(item_vs.add(new_item,
                                    item_vs.scalar_mul(item, -np.uint64(1))))
        return vspace(data).add(data, diff)
    return item, update

def decompose(x, n_channels, patch_size=PATCH_SIZE):
    # Decompose input x into spatial patches
    if isinstance(x, torch.Tensor) is False:
        x = torch.Tensor(x)

    y = x.unfold(1, n_channels, n_channels).unfold(2, *patch_size).unfold(3, *patch_size)
    unfolded_shape = tuple(y.size())
    y = torch.reshape(y, (-1, n_channels, *patch_size))  # (n_patches, n_channels, *patch_size)
    return y.cpu().numpy().astype(np.int32), unfolded_shape
    
def reconstitute(x, original_shape, unfolded_shape, patch_size=PATCH_SIZE):
    # Reconstitute patches into original input
    if isinstance(x, torch.Tensor) is False:
        x = torch.Tensor(x)
        
    B, n_channels, *_ = original_shape

    x_re = torch.reshape(x, 
        (B, 1, unfolded_shape[2], unfolded_shape[3], n_channels, *patch_size))
    x_re = x_re.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(original_shape)

    return x_re.cpu().numpy().astype(np.int32)


def check_argument_shapes(cdf, cdf_length, cdf_offset):
    if (len(cdf.size()) != 2 or cdf.size(1) < 3):
        raise ValueError("'cdf' should be 2-D and cdf.dim_size(1) >= 3: ", cdf.size())

    if (len(cdf_length.size()) != 1 or cdf_length.size(0) != cdf.size(0)):
        raise ValueError("'cdf_length' should be 1-D and its length "
            "should match the number of rows in 'cdf': ", cdf_length.size())

    if (len(cdf_offset.size()) != 1 or cdf_offset.size(0) != cdf.size(0)):
        raise ValueError("'cdf_offset' should be 1-D and its length "
            "should match the number of rows in 'cdf': ", cdf_offset.size())


def ans_compress(symbols, indices, cdf, cdf_length, cdf_offset, coding_shape,
    precision, vectorize=False, block_encode=True):

    if vectorize is True:  # Inputs must be identically shaped

        encoded = entropy_coding.vec_ans_index_encoder(
            symbols=symbols,  # [N, C, H, W]
            indices=indices,  # [N, C, H, W]
            cdf=cdf,  # [n_scales, max_length + 2]
            cdf_length=cdf_length,  # [n_scales]
            cdf_offset=cdf_offset,  # [n_scales]
            precision=precision,
            coding_shape=coding_shape,
        )

    else:
        if block_encode is True:

            encoded = entropy_coding.ans_index_encoder(
                symbols=symbols,  # [N, C, H, W]
                indices=indices,  # [N, C, H, W]
                cdf=cdf,  # [n_scales, max_length + 2]
                cdf_length=cdf_length,  # [n_scales]
                cdf_offset=cdf_offset,  # [n_scales]
                precision=precision,
                coding_shape=coding_shape,
            )
            
        else: 

            encoded = []

            for i in range(symbols.shape[0]):

                coded_string = entropy_coding.ans_index_encoder(
                    symbols=symbols[i],  # [C, H, W]
                    indices=indices[i],  # [C, H, W]
                    cdf=cdf,  # [n_scales, max_length + 2]
                    cdf_length=cdf_length,  # [n_scales]
                    cdf_offset=cdf_offset,  # [n_scales]
                    precision=precision,
                    coding_shape=coding_shape,
                )

                encoded.append(coded_string)

    return encoded


def ans_decompress(encoded, indices, cdf, cdf_length, cdf_offset, coding_shape,
    precision, vectorize=False, block_decode=True):

    if vectorize is True:  # Inputs must be identically shaped

        decoded = entropy_coding.vec_ans_index_decoder(
            encoded,
            indices=indices,
            cdf=cdf,
            cdf_length=cdf_length,
            cdf_offset=cdf_offset,
            precision=precision,
            coding_shape=coding_shape,
        )

    else:
        
        if block_decode is True:

            decoded = entropy_coding.ans_index_decoder(
                encoded,
                indices=indices,  # [N, C, H, W]
                cdf=cdf,  # [n_scales, max_length + 2]
                cdf_length=cdf_length,  # [n_scales]
                cdf_offset=cdf_offset,  # [n_scales]
                precision=precision,
                coding_shape=coding_shape,
            )

        else: 

            decoded = []
            assert len(encoded) == batch_shape, (
                f'Encoded batch dim {len(encoded)} != batch size {batch_shape}')

            for i in range(batch_shape):

                coded_string = entropy_coding.ans_index_decoder(
                    encoded[i],  # [C, H, W]
                    indices=indices[i],  # [C, H, W]
                    cdf=cdf,  # [n_scales, max_length + 2]
                    cdf_length=cdf_length,  # [n_scales]
                    cdf_offset=cdf_offset,  # [n_scales]
                    precision=precision,
                    coding_shape=coding_shape,
                )

                decoded.append(coded_string)

            decoded = np.stack(decoded, axis=0)

    return decoded

def compose(*args):
    """
    :param args: a list of functions
    :return: composition of the functions
    """
    def compose2(f1, f2):
        def composed(*args_c, **kwargs_c):
            return f1(f2(*args_c, **kwargs_c))
        return composed

    return functools.reduce(compose2, args)

def return_list(f):
    """ Can be used to decorate generator functions. """
    return compose(list, f)

def write_coding_shape(shape, fout):
    """
    Write tuple (C,H,W) to file.
    """
    assert len(shape) == 3, shape
    assert shape[0] < 2**16, shape
    assert shape[1] < 2**16, shape
    assert shape[2] < 2**16, shape
    write_bytes(fout, [np.uint16, np.uint16, np.uint16], shape)
    return 6  # number of bytes written

def write_shapes(shape, fout):
    """
    Write tuple (H,W) or (C,H,W) to file.
    """
    for s in shape:
        assert s < 2**16, s

    write_bytes(fout, [np.uint16]*len(shape), shape)
    return 2*len(shape)  # number of bytes written

def read_shapes(fin, shape_len):
    return tuple(map(int, read_bytes(fin, [np.uint16]*shape_len)))

def write_num_bytes_encoded(num_bytes, fout):
    assert num_bytes < 2**32
    write_bytes(fout, [np.uint32], [num_bytes])
    return 4  # number of bytes written

def read_num_bytes_encoded(fin):
    return int(read_bytes(fin, [np.uint32])[0])

def write_bytes(f, ts, xs):
    for t, x in zip(ts, xs):
        f.write(t(x).tobytes())

@return_list
def read_bytes(f, ts):
    for t in ts:
        num_bytes_to_read = t().itemsize
        yield np.frombuffer(f.read(num_bytes_to_read), t, count=1)

def message_to_bytes(f_out, message):
    # Message should be of type np.int32
    # with open(p, 'wb') as f:
    f_out.write(message.tobytes())

def message_from_bytes(f_in, num_bytes, t=np.uint32):
    message = np.frombuffer(f_in.read(num_bytes), t, count=-1)
    return message


def save_compressed_format(compression_output, out_path):
    # Saves compressed output to disk at `out_path`, together with
    # necessary meta-information

    # See format of compressed output in `hyperprior.py`
    entropy_coding_bytes = []

    with open(out_path, 'wb') as f_out:

        # Write meta information
        write_shapes(compression_output.hyperlatent_spatial_shape, f_out)  # (H,W)
        write_shapes(compression_output.spatial_shape, f_out)  # (H,W)
        write_shapes(compression_output.hyper_coding_shape, f_out)  # (C,H,W)
        write_shapes(compression_output.latent_coding_shape, f_out)  # (C,H,W)
        write_shapes([compression_output.batch_shape], f_out)  # (B)
        f_out.write(_MAGIC_VALUE_SEP)

        # Write hyperlatents
        enc_hyperlatents = compression_output.hyperlatents_encoded  # np.uint32
        write_num_bytes_encoded(len(enc_hyperlatents) * 4, f_out)
        message_to_bytes(f_out, enc_hyperlatents)
        f_out.write(_MAGIC_VALUE_SEP)

        # Write latents
        enc_latents = compression_output.latents_encoded
        write_num_bytes_encoded(len(enc_latents) * 4, f_out)
        message_to_bytes(f_out, enc_latents)
        f_out.write(_MAGIC_VALUE_SEP)

    actual_num_bytes = os.path.getsize(out_path)
    actual_bpp = 8. * float(actual_num_bytes) / np.prod(compression_output.spatial_shape)
    try:
        theoretical_bpp = float(compression_output.total_bpp.item())
    except AttributeError:
        theoretical_bpp = float(compression_output.total_bpp)

    return actual_bpp, theoretical_bpp

def load_compressed_format(in_path):
    # Loads necessary meta-information and compressed format from binary
    # format on disk

    with open(in_path, 'rb') as f_in:

        # Read meta information
        hyperlatent_spatial_shape = read_shapes(f_in, 2)
        spatial_shape = read_shapes(f_in, 2)
        hyper_coding_shape = read_shapes(f_in, 3)
        latent_coding_shape = read_shapes(f_in, 3)
        batch_shape = read_shapes(f_in, 1)
        assert f_in.read(4) == _MAGIC_VALUE_SEP  # assert valid file
        
        # Read hyperlatents
        num_bytes = read_num_bytes_encoded(f_in)
        hyperlatents_encoded = message_from_bytes(f_in, num_bytes)
        assert f_in.read(4) == _MAGIC_VALUE_SEP  # assert valid file

        # Read latents
        num_bytes = read_num_bytes_encoded(f_in)
        latents_encoded = message_from_bytes(f_in, num_bytes)
        assert f_in.read(4) == _MAGIC_VALUE_SEP  # assert valid file

    compression_output = CompressionOutput(
        hyperlatents_encoded=hyperlatents_encoded,
        latents_encoded=latents_encoded,
        hyperlatent_spatial_shape=hyperlatent_spatial_shape,  # 2D
        spatial_shape=spatial_shape,  # 2D
        hyper_coding_shape=hyper_coding_shape,  # C,H,W
        latent_coding_shape=latent_coding_shape,  # C,H,W
        batch_shape=batch_shape[0])

    return compression_output

if __name__ == '__main__':

    import random

    quantile = 0.42 + random.randint(0,50)/100

    norm_cdf = lambda x: 0.5 * (1. + torch.erf(x / math.sqrt(2)))
    tails = estimate_tails(norm_cdf, quantile, shape=10)
    print('Normal:')
    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")

    quantile = 0.69 + random.randint(0,31)/100

    norm_cdf = torch.sigmoid
    tails = estimate_tails(norm_cdf, quantile, shape=10)
    print('logistic:')
    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")

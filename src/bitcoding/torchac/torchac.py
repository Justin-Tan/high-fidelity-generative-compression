"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
# TODO some comments needed about [..., -1] == 0


import sys
import torch

try:
    import torchac_backend_gpu as torchac_backend
    CUDA_SUPPORTED = True
except ImportError:
    CUDA_SUPPORTED = False
    # Try importing the cpu version
    try:
        import torchac_backend_cpu as torchac_backend
    except ImportError:
        print('*** ERROR: torchac_backend not found. Please see README.')
        sys.exit(1)


def encode_cdf(cdf, sym):
    """
    :param cdf: CDF as 1HWLp, as int16, on CPU
    :param sym: the symbols to encode, as int16, on CPU
    :return: byte-string, encoding `sym`
    """
    return torchac_backend.encode_cdf(cdf, sym)


def decode_cdf(cdf, input_string):
    """
    :param cdf: CDF as 1HWLp, as int16, on CPU
    :param input_string: byte-string, encoding some symbols `sym`.
    :return: decoded `sym`.
    """
    return torchac_backend.decode_cdf(cdf, input_string)


def encode_logistic_mixture(
        targets, means, log_scales, logit_probs_softmax,  # CDF
        sym):
    """
    NOTE:
        If you compiled torchac_backend with CUDA support, all tensors must be on the GPU!
        If you compiled it *without* CUDA support, they must be on CPU.
    In the following, we use
        Lp: Lp = L+1, where L = number of symbols.
        K: number of mixtures
    :param targets: values of symbols, tensor of length Lp, float32
    :param means: means of mixtures, tensor of shape 1KHW, float32
    :param log_scales: log(scales) of mixtures, tensor of shape 1KHW, float32
    :param logit_probs_softmax: weights of the mixtures (PI), tensorf of shape 1KHW, float32
    :param sym: the symbols to encode
    :return: byte-string, encoding `sym`.
    """
    if CUDA_SUPPORTED:
        return torchac_backend.encode_logistic_mixture(
                targets, means, log_scales, logit_probs_softmax, sym)
    else:
        cdf = _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales)
        return encode_cdf(cdf, sym)


def decode_logistic_mixture(
        targets, means, log_scales, logit_probs_softmax,  # CDF
        input_string):
    """
    NOTE:
        If you compiled torchac_backend with CUDA support, all tensors must be on the GPU!
        If you compiled it *without* CUDA support, they can be either on GPU or CPU.
    In the following, we use
        Lp: Lp = L+1, where L = number of symbols.
        K: number of mixtures
    :param targets: values of symbols, tensor of length Lp, float32
    :param means: means of mixtures, tensor of shape 1KHW, float32
    :param log_scales: log(scales) of mixtures, tensor of shape 1KHW, float32
    :param logit_probs_softmax: weights of the mixtures (PI), tensorf of shape 1KHW, float32
    :param input_string: byte-string, encoding some symbols `sym`.
    :return: decoded `sym`.
    """
    if CUDA_SUPPORTED:
        return torchac_backend.decode_logistic_mixture(
                targets, means, log_scales, logit_probs_softmax, input_string)
    else:
        cdf = _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales)
        return decode_cdf(cdf, input_string)


def _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales):
    cdf_float = _get_C_cur_weighted(logit_probs_softmax, targets, means, log_scales)
    cdf = _renorm_cast_cdf_(cdf_float, precision=16)
    cdf = cdf.cpu()
    return cdf


def _get_C_cur_weighted(logit_probs_softmax_c, targets, means_c, log_scales_c):
    C_cur = _get_C_cur(targets, means_c, log_scales_c)  # NKHWL
    C_cur = C_cur.mul(logit_probs_softmax_c.unsqueeze(-1)).sum(1)  # NHWL
    return C_cur


def _get_C_cur(targets, means_c, log_scales_c):  # NKHWL
    """
    :param targets: Lp floats
    :param means_c: NKHW
    :param log_scales_c: NKHW
    :return:
    """
    # NKHW1
    inv_stdv = torch.exp(-log_scales_c).unsqueeze(-1)
    # NKHWL'
    centered_targets = (targets - means_c.unsqueeze(-1))
    # NKHWL'
    cdf = centered_targets.mul(inv_stdv).sigmoid()  # sigma' * (x - mu)
    return cdf


def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom
from src.helpers import maths

MIN_SCALE = 0.11
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e4
TAIL_MASS = 2**(-8)

PRECISION_P = 16  # Precision of rANS coder

# TODO: Unit tests

lower_bound_toward = maths.LowerBoundToward.apply
    
class ContinuousEntropyModel(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for pre-computation of integer probability tables for use in entropy coding.
    """
    def __init__(self, distribution, likelihood_bound=MIN_LIKELIHOOD, tail_mass=TAIL_MASS,
        precision=PRECISION_P):
        """
        The layer assumes that the input tensor is at least 2D, with a batch dimension
        at the beginning and a channel dimension, specified by subclassing this layer. 
        The layer trains an independent probability density model for each 'channel',
        but assumes that across all other dimensions, the inputs are i.i.d. (independent
        and identically distributed).

        Parameters:
            distribution: Distribution with CDF / quantile / likelihood methods

        Note:
        The batch dimensions are indexes into independent, non-identical parameterizations 
        of this distribution - [B, n_channels], where B usually = 1.
        (Dimensions which are not assumed i.i.d.)
        """

        super(ContinuousEntropyModel, self).__init__()

        self.distribution = distribution
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)
        self.precision = int(precision)

    def quantize_st(self, inputs, offsets=None):
        # Ignore rounding in backward pass
        values = inputs

        if offsets is not None:
            offsets = offsets.to(values)
            values = values - offsets

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if offsets is not None:
            values = values + offsets

        return values

    def dequantize(self, x, offsets=None):

        if offsets is not None:
            values = x.type_as(offsets)
            values = values + offsets
        else:
            values = x.to(torch.float32)

        return values

    @abc.abstractmethod
    def build_tables(self, **kwargs):
        pass



if __name__ == '__main__':

    print('Hi!')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import math, normalization
from models import network

EPS = 1e-8

class Uniform(nn.Module):

    def __init__(self, low=0., high=1.):
        super(Uniform, self).__init__()

        self.low = low
        self.high = high

    def sample(self, shape, **kwargs):
        rand = torch.rand(shape)#, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_density(self, x, low=0., high=1., EPS=1e-3, **kwargs):
        low = torch.Tensor((low-EPS,)).to(x)
        high = torch.Tensor((high+EPS,)).to(x)
        lb = low.le(x).type_as(low)
        ub = high.gt(x).type_as(low)
        return torch.log(EPS + (1.-EPS) * lb.mul(ub)) - torch.log(high - low)

class StudentT(nn.Module):

    def __init__(self, df=1., loc=0., scale=1.):
        super(StudentT, self).__init__()

        self.df = df
        self.loc = loc
        self.scale = scale
        self.dist = torch.distributions.studentT.StudentT(self.df, self.loc, self.scale)

    def sample(self, shape, df=None, loc=None, scale=None, **kwargs):

        if all(p is not None for p in [df, loc, scale]):
            # Instantiate new distributions object
            st = torch.distributions.studentT.StudentT(df, loc, scale)
            return st.rsample(shape)
        elif all(p is None for p in [df, loc, scale]):
            # use distributions object defined in constructor
            return self.dist.rsample(shape)
        else:
            raise ValueError('Parameters must be fully (un)specified!')

    def reparameterize(self, shape, df=None, loc=None, scale=None, **kwargs):
        
        return self.sample(shape, df, loc, scale, **kwargs)

    def log_density(self, x, df=None, loc=None, scale=None, **kwargs):
        if all(p is not None for p in [df, loc, scale]):
            # Instantiate new distributions object
            st = torch.distributions.studentT.StudentT(df, loc, scale)
            return st.log_prob(x)
        elif all(p is None for p in [df, loc, scale]):
            # use distributions object defined in constructor
            return self.dist.log_prob(x)
        else:
            raise ValueError('Parameters must be fully (un)specified!')

class Normal(nn.Module):

    """
    Normal distribution with diagonal covariance
    """
    def __init__(self, mu=0., logvar=1.):
        super(Normal, self).__init__()

        self.mu = torch.Tensor([mu])
        self.logvar = torch.Tensor([logvar])

    def sample(self, shape, mu, logvar):
        """
        Sample from N(mu(x), Sigma(x)) as 
        z ~ mu + Cholesky(Sigma(x)) * eps
        eps ~ N(0,I_n)
        
        The variance is restricted to be diagonal,
        so Cholesky(...) -> sqrt(...)
        """
        sigma_sqrt = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma_sqrt)
        return mu + sigma_sqrt * epsilon

    def log_density(self, x, mu, logvar):
        """
        First argument of params is location parameter,
        second argument is scale parameter
        """
        return math.log_density_gaussian(x, mu, logvar)

    def NLL(self, params, sample_params=None):
        """
        Analytically computes negative log-likelihood 
        E_N(mu_2, var_2) [- log N(mu_1, var_1)]
        If mu_2, and var_2 are not provided, defaults to entropy.
        """
        mu, logvar = params

        if sample_params is not None:
            sample_mu, sample_logvar = sample_params
        else:
            sample_mu, sample_logvar = mu, logvar

        c = self.normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)


class Bernoulli(nn.Module):
    """
    Bernoulli distribution. Probability given by sigmoid of input parameter.
    For natural image data each pixel is modelled as independent Bernoulli
    """
    def __init__(self, theta=0.5):
        super(Bernoulli, self).__init__()

        self.theta = torch.Tensor([theta])

    def sample(self, theta):
        """
        """
        raise NotImplementedError


    def log_density(self, x, params=None):
        """ x, params \in [0,1] """
        log_px = F.binary_cross_entropy(params, x, reduction="sum")
        return log_px



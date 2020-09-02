import torch
import numpy as np
import scipy.stats

def pmf_to_quantized_cdf(pmf, precision, careful=True):
    """
    Based on https://github.com/rygorous/ryg_rans/blob/master/main64.cpp

    TODO: port to C++

    Convert PMF to quantized CDF. For entropy encoders and decoders to have the same 
    quantized CDF on different platforms, the quantized CDF should be produced once 
    and saved, then the saved quantized CDF should be used everywhere.

    After quantization, if PMF does not sum to 2^precision, then some values of PMF 
    are increased or decreased to adjust the sum to equal to 2^precision.

    Note that the input PMF is pre-quantization. The input PMF is not normalized by 
    this op prior to quantization. Therefore the user is responsible for normalizing 
    PMF if necessary.
    """

    assert precision >= 8, 'Increase precision, $p \in [8,32]$'
    assert pmf.shape[-1] >= 2, 'pmf shape should be at least 2 in last axis'
    assert torch.all(pmf >= 0.).item(), 'PMF must have all non-negative values'
    assert torch.logical_not(torch.all(torch.isnan(pmf))).item(), 'PMF contains NaNs!'

    target_total = 1 << precision
    pmf_shape = pmf.shape
    cdf = torch.zeros(pmf_shape[0] + 1)

    cdf[0] = 0
    cdf[1:] = torch.cumsum(pmf, dim=0)
    empirical_total = cdf[-1]

    # Normalize CDF to desired precision
    cdf = torch.round(cdf * target_total / empirical_total).to(torch.int64)

    # If we nuked any non-0 frequency symbol to 0, we need to steal
    # the range to make the frequency nonzero from elsewhere.

    for i in range(cdf.size(0) - 1):

        if (cdf[i] == cdf[i + 1]):
            # Steal frequency from low-frequency symbols
            best_freq = target_total + 1
            best_steal = -1

            for j in range(cdf.size(0) - 1):
                freq = cdf[j + 1] - cdf[j]

                if (freq > 1 and freq < best_freq):
                    best_freq = freq
                    best_steal = j

            assert best_steal != -1

            if (best_steal < i):
                for j in range(best_steal + 1, i + 1):
                    cdf[j] -= 1
            else:
                assert best_steal > i
                for j in range(i + 1, best_steal + 1):
                    cdf[j] += 1

    assert cdf[0] == 0, 'Error in CDF normalization'
    assert cdf[-1] == 1 << precision, 'Error in CDF normalization'

    if careful is True:
        for i in range(cdf.size(0)-1):
            assert cdf[i+1] >= cdf[i], 'CDF function is not monotonic!'

    return cdf


class LowerBoundIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, lower_bound):
        ctx.lower_bound = lower_bound
        return torch.clamp(tensor, lower_bound)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class LowerBoundToward(torch.autograd.Function):
    """
    Assumes output shape is identical to input shape.
    """
    @staticmethod
    def forward(ctx, tensor, lower_bound):
        # lower_bound:  Scalar float.
        ctx.mask = tensor.ge(lower_bound)
        return torch.clamp(tensor, lower_bound)

    @staticmethod
    def backward(ctx, grad_output):
        gate = torch.logical_or(ctx.mask, grad_output.lt(0.)).type(grad_output.dtype)
        return grad_output * gate, None

def standardized_CDF_gaussian(value):
    # Gaussian
    # return 0.5 * (1. + torch.erf(value/ np.sqrt(2)))
    return 0.5 * torch.erfc(value * (-1./np.sqrt(2)))

def standardized_CDF_logistic(value):
    # Logistic
    return torch.sigmoid(value)

def standardized_quantile_gaussian(quantile):
    return scipy.stats.norm.ppf(quantile)

def standardized_quantile_logistic(quantile):
    return scipy.stats.logistic.ppf(quantile)

def quantile_gaussian(quantile, mean, scale):
    return scipy.stats.norm.ppf(quantile, loc=mean, scale=scale)

def quantile_logistic(quantile, mean, scale):
    return scipy.stats.logistic.ppf(quantile, loc=mean, scale=scale)

def gaussian_entropy(D, logvar):
    """
    Entropy of a Gaussian distribution with 'D' dimensions and heteroscedastic log variance 'logvar'
    Parameters
    ----------
    D:      integer
            Dimension of Gaussian distribution
    logvar: torch.Tensor 
            logvar for each example in batch, [batch_size, D]
    """
    h = 0.5 * (D * (np.log(2.0 * np.pi) + 1) + torch.sum(logvar, dim=1))

def log_density_gaussian(x, mu=None, logvar=None):
    """
    Calculates log density of a Gaussian.
    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.

    Returns:
    log_density: [B, latent_dim]
    """
    if mu is None and logvar is None:
        mu = torch.zeros_like(x)
        logvar = torch.zeros_like(x)
        
    normalization = -0.5 * (np.log(2 * np.pi) + logvar)
    # Logvar should be above exp(-5)
    inv_var = torch.exp(torch.min(-logvar, torch.ones_like(logvar)*5))
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)

    return log_density

def gaussian_sample(mu, logvar):
    """
    Sample from N(mu, Sigma): 
    z ~ mu + Cholesky(Sigma(x)) * eps
    eps ~ N(0,I_n)
    
    The variance is restricted to be diagonal,
    so Cholesky(...) -> sqrt(...)
    Parameters
    ----------
    mu     : torch.Tensor
        Location parameter of Gaussian. (B, D)
    logvar : torch.Tensor
        Log of variance parameter of Gaussian. (B, D)
    """
    sigma = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon
        

def kl_divergence_q_prior_normal(mu, logvar):
    """
    Returns KL-divergence between the variational posterior
    $q_{\phi}(z|x)$ and the isotropic Gaussian prior $p(z)$.
    
    If the variational posterior is taken to be normal with 
    diagonal covariance. Then:
    $ D_{KL}(q_{\phi(z|x)}||p(z)) = -1/2 * \sum_j (1 + log \sigma_j^2 - \mu_j^2 - \sigma_j^2) $
    """
    
    assert mu.shape == logvar.shape, 'Mean and log-variance must share shape (batch, latent_dim)'
    latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).sum(dim=1)
    return latent_kl


def matrix_log_density_gaussian(x, mu, logvar):
    """
    Calculates log density of a Gaussian for all combination of batch pairs of
    `x` and `mu`. i.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch

    Returns:
    log_density_matrix: [B,B,latent_dim]
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)

    return log_density_gaussian(x, mu, logvar)


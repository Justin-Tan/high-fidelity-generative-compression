import torch
import numpy as np

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

def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix
    Parameters
    ----------
    batch_size: int
        number of training images in the batch
    dataset_size: int
        number of training images in the dataset

    Returns:
    log_W: Matrix of importance weights for estimating
    log_qz, etc. according to (S6) of [1], repeat for
    each x* (n*) in batch.
    Shape [B,B]

    [1]: https://arxiv.org/pdf/1802.04942.pdf
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out

def logit(x, eps=1e-6):
    return torch.log(x + eps) - torch.log(1. - x + eps)

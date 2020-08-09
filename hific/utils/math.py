import torch
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve

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
    # inv_var = torch.exp(-logvar)
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

def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
    offset = 0.5
    stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.

    Raises:
    RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                        img1.shape, img2.shape)
    if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                        img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs

def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
    weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.

    Returns:
    MS-SSIM score between `img1` and `img2`.

    Raises:
    RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                        img1.shape, img2.shape)
    if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                        img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in xrange(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
            (mssim[levels-1] ** weights[levels-1]))
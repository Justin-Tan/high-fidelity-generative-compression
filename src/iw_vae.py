import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def reparameterize_continuous(mu, sigma):
    """
    Sample from N(mu(x), Sigma(x)) as 
    z | x ~ mu + Cholesky(Sigma(x)) * eps
    eps ~ N(0,I_n)
    
    The variance is restricted to be diagonal,
    so Cholesky(...) -> sqrt(...)
    Parameters
    ----------
    mu     : torch.Tensor
        Location parameter of Gaussian. (B, latent_dim)

    logvar : torch.Tensor
        Log of variance parameter of Gaussian. (B, latent_dim)

    """
    # sample = self.training
    sample = True
    if sample is True:
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
    else:
        # Reconstruction, return mean
        return mu
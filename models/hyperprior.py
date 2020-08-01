import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from models import network
from utils import math, distributions, initialization, helpers

class Hyperprior(nn.Module):
    
    def __init__(self, bottleneck_capacity=220):
        """
        Hyperprior. Introduces probabilistic model over latents.
        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """
        super(Hyperprior, self).__init__()
        
        self.bottleneck_capacity =bottleneck_capacity

        encoder = network.HyperpriorEncoder
        decoder = network.HyperpriorDecoder

        self.encoder = encoder(C=bottleneck_capacity)
        self.decoder = decoder(C=bottleneck_capacity)
        self.reset_parameters()
            
    def reset_parameters(self):
        self.apply(initialization.weights_init)

    def reparameterize_continuous(self, mu, logvar):
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
            sigma = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(sigma)
            return mu + sigma * epsilon
        else:
            # Reconstruction, return mean
            print('MEAN ONLY')
            return mu

    def log_px_estimate(self, x, x_stats, latent_sample, latent_stats, **kwargs):
        log_pxCz = math.log_density_gaussian(x, mu=x_stats['mu'], logvar=x_stats['logvar']).sum(1, keepdim=False)
        KL_qzCx_pz = math.kl_divergence_q_prior_normal(*latent_stats)
        return log_pxCz - KL_qzCx_pz
    
    def forward(self, x, **kwargs):
        latent_stats = self.encoder(x)['continuous']
        latent_sample = self.reparameterize_continuous(*latent_stats)
        x_stats = self.decoder(latent_sample)
        
        return x_stats, latent_sample, latent_stats


class UnivariateDensity(nn.Module):
    """
    Probability model for hyper-latents
    """

    def __init__(self, init_scale=10, filters=(3, 3, 3), **kwargs):
        """
        init_scale: Scaling factor determining the initial width of the
                    probability densities.
        filters:    Number of filters at each layer
                    of the density model.
        """
        super(UnivariateDensity, self).__init__()
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from models import network
from utils import math, distributions, initialization, helpers

class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim=8):
        """
        Lightweight VAE implementation for density estimation
        TODO image data
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = np.prod(self.input_dim)
        self.latent_dim = latent_dim
        self.latent_spec = {'continuous': self.latent_dim}

        self.prior = distributions.Normal()
        self.z_dist = distributions.Normal()
        self.x_dist = distributions.Normal()  # distributions.Bernoulli()

        encoder = network.ToyEncoder
        decoder = network.ToyDecoder

        self.encoder = encoder(input_dim=self.input_dim, latent_spec=self.latent_spec, hidden_dim=self.hidden_dim)
        self.decoder = decoder(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,
            output_dim=self.output_dim)
        self.reset_parameters()

        print('Using prior:', self.prior)
        print('Using likelihood p(x|z):', self.x_dist)
        print('Using approximate posterior p(z|x): Diagonal-covariance Gaussian')
            
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
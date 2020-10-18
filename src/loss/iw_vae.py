import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.helpers import maths

lower_bound_toward = maths.LowerBoundToward.apply

class IWAE(nn.Module):

    def __init__(self, bottleneck_capacity, latent_prob_model, hyperlatent_prob_model, hyperlatent_inference_net, 
        synthesis_net, scale_lower_bound, num_i_samples=4, use_dreg=False, **kwargs):
        
        super(IWAE, self).__init__()

        self.synthesis_net = synthesis_net
        self.latent_prob_model = latent_prob_model
        self.hyperlatent_prob_model = hyperlatent_prob_model
        self.hyperlatent_inference_net = hyperlatent_inference_net
        
        self.use_dreg = True
        self.num_i_samples = num_i_samples
        self.scale_lower_bound = scale_lower_bound
        self.bottleneck_capacity = bottleneck_capacity

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
            return mu


    def _estimate_entropy_log(self, log_likelihood, spatial_shape):

        quotient = -np.log(2.)
        batch_size = log_likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return bpp

    def get_importance_samples(self, latents, latent_stats, hyperlatent_sample, hyperlatent_stats):
        """
        Tighten lower bound by reducing variance of marginal likelihood
        estimator through the sample mean.
        """

        EPS = 1e-9  
        # [n*B, C_y, H_y, W_y]
        B, C_y, H_y, W_y = latents.size()
        latents = torch.repeat_interleave(latents, self.num_i_samples, dim=0)

        # [n*B, C_y, H_y, W_y]
        log_pz = torch.log(self.hyperlatent_prob_model(hyperlatent_sample) + EPS)#.sum(dim=(1,2,3), keepdim=True)
        log_qzCy = maths.log_density_gaussian(hyperlatent_sample, *hyperlatent_stats)#.sum(dim=(1,2,3), keepdim=True)
        log_pyCz = torch.log(self.latent_prob_model(latents, *latent_stats) + EPS)

        log_iw = log_pyCz + (log_pz - log_qzCy).sum(dim=(1,2,3), keepdim=True)/(C_y * H_y * W_y)
        # log_iw_scalar = log_pyCz.sum(dim=(1,2,3), keepdim=True) + log_pz.sum(dim=(1,2,3), keepdim=True) - log_qzCy.sum(dim=(1,2,3), keepdim=True)
    
        log_iw = log_iw.reshape(B, self.num_i_samples, C_y, H_y, W_y)  # [B, n, C_y, H_y, W_y]
        # print('after reshape', log_iw.shape)
        return log_iw #, log_iw_scalar

    def _marginal_estimate(self, latents, latent_stats, hyperlatent_sample, hyperlatent_stats, **kwargs):
    
        EPS = 1e-9  
        # [n*B, C_y, H_y, W_y]
        B, C_y, H_y, W_y = latents.size()
        latents = torch.repeat_interleave(latents, self.num_i_samples, dim=0)

        # [n*B, C_y, H_y, W_y] -> [n*B]
        log_pz = torch.log(self.hyperlatent_prob_model(hyperlatent_sample) + EPS).sum(dim=(1,2,3), keepdim=True)
        log_qzCy = maths.log_density_gaussian(hyperlatent_sample, *hyperlatent_stats).sum(dim=(1,2,3), keepdim=True)
        log_pyCz = torch.log(self.latent_prob_model(latents, *latent_stats) + EPS).sum(dim=(1,2,3), keepdim=True)
        #print(log_pyCz.shape)
        #print(torch.mean(log_pyCz)/(-np.log(2) * 256 * 256))

        log_iw = log_pyCz + (log_pz - log_qzCy)
        log_iw = log_iw.reshape(B, self.num_i_samples) # [B, n]

        # log_iw = self.get_importance_samples(latents, latent_stats, hyperlatent_sample, hyperlatent_stats)
        iwelbo = torch.logsumexp(log_iw, dim=1) - np.log(self.num_i_samples)  # [B]

        return iwelbo

    def _marginal_dreg_estimate(self, latents, latent_stats, hyperlatent_sample, hyperlatent_stats, **kwargs):
        """
        Doubly reparameterized gradient estimator.
        """
        EPS = 1e-9  
        # [n*B, C_y, H_y, W_y]
        B, C_y, H_y, W_y = latents.size()
        latents = torch.repeat_interleave(latents, self.num_i_samples, dim=0)

        # [n*B, C_y, H_y, W_y] - > [n*B]
        # log_pz = torch.log(self.hyperlatent_prob_model(hyperlatent_sample) + EPS).sum(dim=(1,2,3), keepdim=True)
        log_pz = self.hyperlatent_prob_model(hyperlatent_sample).sum(dim=(1,2,3), keepdim=True)

        # Override gradients of inference network
        log_qzCy = maths.log_density_gaussian(hyperlatent_sample, *[stat.detach() for stat in hyperlatent_stats]).sum(dim=(1,2,3), keepdim=True)
        log_pyCz = torch.log(self.latent_prob_model(latents, *latent_stats) + EPS).sum(dim=(1,2,3), keepdim=True)

        log_iw_stop_phi = log_pyCz + (log_pz - log_qzCy)
        log_iw_stop_phi = log_iw_stop_phi.reshape(B, self.num_i_samples) # [B, n]

        with torch.no_grad():
            normalized_weights = torch.exp(log_iw_stop_phi - torch.logsumexp(log_iw_stop_phi, dim=1, keepdim=True)).detach()
            # normalized_weights = F.softmax(log_iw_stop_phi, dim=1).detach()
            if hyperlatent_sample.requires_grad:
                hyperlatent_sample.register_hook(lambda grad: torch.reshape(normalized_weights, (B * self.num_i_samples, 1, 1, 1))  * grad)
                # hyperlatent_sample.register_hook(lambda grad: print(grad.shape))

        iw_dreg = torch.sum(normalized_weights * log_iw_stop_phi, dim=1)  # [B]

        return iw_dreg

    def marginal_estimate(self, latents, latent_stats, hyperlatent_sample, hyperlatent_stats, **kwargs):
        
        if self.use_dreg is True:
            return self._marginal_dreg_estimate(latents, latent_stats, hyperlatent_sample, hyperlatent_stats, **kwargs)
        else:
            return self._marginal_estimate(latents, latent_stats, hyperlatent_sample, hyperlatent_stats, **kwargs)

    def amortized_inference(self, latents, hyperlatent_stats, num_i_samples):

        hyperlatent_mu, hyperlatent_logvar = hyperlatent_stats  # [B, C_z, H_z, W_z]

        # Sample num_i_samples for each batch element
        mu_z = torch.repeat_interleave(hyperlatent_mu, self.num_i_samples, dim=0)
        logvar_z = torch.repeat_interleave(hyperlatent_logvar, self.num_i_samples, dim=0)
    
        hyperlatent_sample = self.reparameterize_continuous(mu=mu_z, logvar=logvar_z)
        latent_params = self.synthesis_net(hyperlatent_sample)  # [n*B, C_y, H_y, W_y]
        latent_means, latent_scales = torch.split(latent_params, self.bottleneck_capacity, dim=1)   
        latent_scales = lower_bound_toward(F.softplus(latent_scales), self.scale_lower_bound)
        latent_stats = (latent_means, latent_scales)

        return latent_stats, hyperlatent_sample, (mu_z, logvar_z)

    def forward(self, latents, hyperlatent_stats, num_i_samples=None, **kwargs):

        if num_i_samples is not None:
            print('Using {} importance samples'.format(num_i_samples))
            self.num_i_samples = num_i_samples

        latent_stats, hyperlatent_sample, hyperlatent_stats = self.amortized_inference(latents, hyperlatent_stats,
            num_i_samples=self.num_i_samples)
        iwelbo = self.marginal_estimate(latents, latent_stats, hyperlatent_sample, hyperlatent_stats)

        return iwelbo

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from models import network
from utils import math, distributions, initialization, helpers


class HyperpriorDensity(nn.Module):
    """
    Probability model for hyper-latents z. Based on Sec. 6.1. of [1]

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).
    """

    def __init__(self, n_channels, init_scale=10, filters=(3, 3, 3), min_likelihood=1e-9, **kwargs):
        """
        init_scale: Scaling factor determining the initial width of the
                    probability densities.
        filters:    Number of filters at each layer < K
                    of the density model. Default K=4 layers.
        """
        super(HyperpriorDensity, self).__init__()
        
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = 16.

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Define univariate density model 
        for k in range(len(self.filters)+1):
            
            # Weights
            H_init = np.log(np.expm1(1 / scale / filters[k + 1]))
            H_k = nn.Parameter(torch.ones((n_channels, filters[k+1], filters[k])))  # apply softmax for non-negativity
            torch.nn.init.constant_(H_k, H_init)
            self.register_parameter('H_{}'.format(k), H_k)

            # Scale factors
            a_k = nn.Parameter(torch.zeros((n_channels, filters[k+1], 1)))
            self.register_parameter('a_{}'.format(k), a_k)

            # Biases
            b_k = nn.Parameter(torch.zeros((n_channels, filters[k+1], 1)))
            torch.nn.init.uniform_(b_k, -0.5, 0.5)
            self.register_parameter('b_{}'.format(k), b_k)

    def cdf_logits(self, x, update_parameters=True):
        """
        Evaluate logits of the cumulative densities. 
        Independent density model for each channel.

        x:  The values at which to evaluate the cumulative densities.
            torch.Tensor - shape `(C, 1, *)`.
        """
        logits = x

        for k in range(len(self.filters)+1):
            H_k = getattr(self, 'H_{}'.format(str(k)))  # Weight
            a_k = getattr(self, 'a_{}'.format(str(k)))  # Scale
            b_k = getattr(self, 'b_{}'.format(str(k)))  # Bias

            if update_parameters is False:
                H_k, a_k, b_k = H_k.detach(), a_k.detach(), b_k.detach()

            logits = torch.bmm(F.softplus(H_k), logits)  # [C,filters[k+1],B]
            logits += b_k
            logits += torch.tanh(a_k) * torch.tanh(logits)

        return logits

    def forward(self, x):
        # Expected input: (N,C,H,W)
        # Convert latents to (C,1,*) format
        latents = x
        N, C, H, W = latents.size()
        latents = latents.permute(1,0,2,3)
        latents = torch.reshape(latents, (C,1,-1))

        cdf_upper = self.cdf_logits(latents + 0.5)
        cdf_lower = self.cdf_logits(latents - 0.5)

        # Numerical stability using some sigmoid identities
        # to avoid subtraction of two numbers close to 1
        sign = -torch.sign(cdf_upper + cdf_lower).detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * cdf_upper) - torch.sigmoid(sign * cdf_lower))

        # Reshape to (N,C,H,W)
        likelihood = torch.reshape(likelihood, (C,N,H,W))
        likelihood = likelihood.permute(1,0,2,3)

        return torch.clamp(likelihood, min=self.min_likelihood, max=self.max_likelihood)



class Hyperprior(nn.Module):
    
    def __init__(self, bottleneck_capacity=220):
        """
        Hyperprior. Introduces probabilistic model over latents.
        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """
        super(Hyperprior, self).__init__()
        
        self.bottleneck_capacity =bottleneck_capacity

        analysis_net = network.HyperpriorAnalysis
        synthesis_net = network.HyperpriorSynthesis

        self.analysis_net = analysis_net(C=bottleneck_capacity)

        # TODO: Combine scale, loc into single network
        self.synthesis_mu = synthesis_net(C=bottleneck_capacity)
        self.synthesis_logvar = synthesis_net(C=bottleneck_capacity)

        self.hyperlatent_likelihood = HyperpriorDensity(n_channels=bottleneck_capacity)
    
    def _quantize(self, x, mode='noise'):
        """
        training:   Boolean. If true, returns continuous relaxation of hard
                    quantization through additive uniform noise channel.
                    Otherwise perform actual quantization (through rounding).
        """

        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise
        elif mode == 'quantize'
            x = torch.round(x)
        else:
            raise NotImplementedError
        
        return x

    def _estimate_entropy(self, x, likelihood):
        # x: (N,C,H,W)
        EPS = 1e-6
        input_shape = x.size()
        quotient = -np.log(2.)

        batch_size = input_shape[0]
        n_pixels = input_shape[2] * input_shape[3]

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp


    def forward(self, latents, **kwargs):

        hyperlatents = self.analysis_net(latents)
        
        # Mismatch b/w continuous and discrete cases?
        # Differential entropy
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        noisy_hyperlatent_likelihood = self.hyperlatent_likelihood(noisy_hyperlatents)
        noisy_bits, noisy_bpp = self._estimate_entropy(hyperlatents, 
            noisy_hyperlatent_likelihood)

        # Discrete entropy
        quantized_hyperlatents = self._quantize(hyperlatents, mode='quantize')
        quantized_hyperlatent_likelihood = self.hyperlatent_likelihood(quantized_hyperlatents)
        quantized_bits, quantized_bpp = self._estimate_entropy(quantized_hyperlatents, 
            quantized_hyperlatent_likelihood)

        if self.training is True:
            hyperlatents_decoded = noisy_hyperlatents
        else:
            hyperlatents_decoded = quantized_hyperlatents

        
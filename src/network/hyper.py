import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers import maths
lower_bound_toward = maths.LowerBoundToward.apply

def get_num_DLMM_channels(C, K=4, params=['mu','scale','mix']):
    """
    C:  Channels of latent representation (L3C uses 5).
    K:  Number of mixture coefficients.
    """
    return C * K * len(params)

def get_num_mixtures(K_agg, C, params=['mu','scale','mix']):
    return K_agg // (len(params) * C)

def unpack_likelihood_params(x, conv_out, log_scales_min):
    
    N, C, H, W = x.shape
    K_agg = conv_out.shape[1]

    K = get_num_mixtures(K_agg, C)

    # For each channel: K pi / K mu / K sigma 
    conv_out = conv_out.reshape(N, 3, C, K, H, W)
    logit_pis = conv_out[:, 0, ...]
    means = conv_out[:, 1, ...]
    log_scales = conv_out[:, 2, ...]
    log_scales = lower_bound_toward(log_scales, log_scales_min)
    x = x.reshape(N, C, 1, H, W)

    return x, (logit_pis, means, log_scales), K
    

class HyperpriorAnalysis(nn.Module):
    """
    Hyperprior 'analysis model' as proposed in [1]. 

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of input channels
    """
    def __init__(self, C=220, N=320, activation='relu'):
        super(HyperpriorAnalysis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.activation = getattr(F, activation)
        self.n_downsampling_layers = 2

        self.conv1 = nn.Conv2d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv2d(N, N, **cnn_kwargs)

    def forward(self, x):
        
        # x = torch.abs(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x


class HyperpriorSynthesis(nn.Module):
    """
    Hyperprior 'synthesis model' as proposed in [1]. Outputs 
    distribution parameters of input latents.

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of output channels
    """
    def __init__(self, C=220, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(N, C, kernel_size=3, stride=1, padding=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class HyperpriorSynthesisDLMM(nn.Module):
    """
    Outputs distribution parameters of input latents, conditional on 
    hyperlatents, assuming a discrete logistic mixture model.

    C:  Number of output channels
    """
    def __init__(self, C=64, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesisDLMM, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(N, C, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(C, get_num_DLMM_channels(C), kernel_size=1, stride=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        x = self.conv_out(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

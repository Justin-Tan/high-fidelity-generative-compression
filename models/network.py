import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from utils import normalization

class hific_encoder(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=16):

        """ 
        Encoder with convolutional architecture proposed in [1].
        Project image x ([C_in,256,256]) into a feature map of size W/16 x H/16 x C
    
        Arguments:
        image_dims:  Dimensions of input image, (C_in,H,W)
        batch_size: Number of instances per minibatch
        C:          Bottleneck depth, controls bits-per-pixel
                    C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(hific_encoder, self).__init__()
        
        im_channels = self.image_dims[0]
        kernel_dim = 3
        hidden_channels = 64
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        assert input_dim == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=0)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        self.activation = getattr(F, activation)  # (leaky_relu, relu, elu)
        
        if channel_norm is True:
            interlayer_norm = normalization.ChannelNorm2D_wrap
        else:
            interlayer_norm = normalization.InstanceNorm2D_wrap

        self.conv1 = nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1)
        self.norm1 = interlayer_norm((batch_size, filters[0], H1, W1), **norm_kwargs)

        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs)
        self.norm2 = interlayer_norm((batch_size, filters[1], H2, W2), **norm_kwargs)

        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs)
        self.norm3 = interlayer_norm((batch_size, filters[2], H3, W3), **norm_kwargs)

        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs)
        self.norm4 = interlayer_norm((batch_size, filters[3], H4, W4), **norm_kwargs)

        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs)
        self.norm5 = interlayer_norm((batch_size, filters[4], H5, W5), **norm_kwargs)

        # Project channels onto space w/ dimension C
        # Feature maps have dimension W/16 x H/16 x C
        self.conv_out = nn.Conv2d(filters[4], C, kernel_dim, stride=1, **cnn_kwargs)
        
                
    def forward(self, x):
        
        batch_size = x.size(0)
    
        x = self.activation(self.conv1(x))
        x = self.norm1(x)
        x = self.activation(self.conv2(x))
        x = self.norm2(x)
        x = self.activation(self.conv3(x))
        x = self.norm3(x)
        x = self.activation(self.conv4(x))
        x = self.norm4(x)
        x = self.activation(self.conv5(x))
        x = self.norm5(x)
        
        out = self.conv_out(x)
        
        # Reshape for quantization
        out = out.view((batch_size, -1))
        
        return out


"""
TEMPLATES
=====================
Encoders / Decoders
=====================
"""

class EncoderVAE_conv(nn.Module):
    def __init__(self, input_dim, activation='relu', latent_spec=None, hidden_dim=256):
        """ 
        Gaussian encoder $q_{\phi}(z|x)$ with convolutional 
        architecture proposed in [1].
        
        The mean and log-variance of each latent dimension is 
        parameterized by the encoder. $z$ can be later sampled 
        using the reparameterization trick.
        
        [1] Locatello et. al., "Challenging Common Assumptions
        in the Unsupervised Learning of Disentangled 
        Representations", arXiv:1811.12359 (2018).
        """
        
        super(EncoderVAE_conv, self).__init__()
        
        self.latent_dim_continuous = latent_spec['continuous']
        self.input_dim = input_dim
        im_channels = self.input_dim[0]
        kernel_dim = 4
        hidden_channels = 64
        n_ch1 = 32
        n_ch2 = 64
        cnn_kwargs = dict(stride=2, padding=1)
        out_conv_shape = (hidden_channels, kernel_dim, kernel_dim)
        # (leaky_relu, relu, elu)
        self.activation = getattr(F, activation)
        
        self.conv1 = nn.Conv2d(im_channels, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv2 = nn.Conv2d(n_ch1, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv3 = nn.Conv2d(n_ch1, n_ch2, kernel_dim, **cnn_kwargs)
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            self.conv4 = nn.Conv2d(n_ch2, n_ch2, kernel_dim, **cnn_kwargs)
            
        self.dense1 = nn.Linear(np.product(out_conv_shape), hidden_dim)    
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dense_mu = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.dense_logvar = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.is_discrete = ('discrete' in latent_spec.keys())


        if self.is_discrete is True:
            # Specify parameters of categorical distribution
            self.discrete_latents = latent_spec['discrete']
            dense_alphas = [nn.Linear(hidden_dim, alpha_dim) for alpha_dim in self.discrete_latents]
            self.dense_alphas = nn.ModuleList(dense_alphas)
            
    def forward(self, x):
        
        # Holds parameters of latent distribution
        # Divided into continuous and discrete dims
        latent_dist = {'continuous': None}

        batch_size = x.size(0)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            x = self.activation(self.conv4(x))
            
        x = x.view((batch_size, -1))
        x = self.activation(self.dense1(x))
        # x = activation(self.dense2(x))
        
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        latent_dist['continuous'] = [mu, logvar]
        latent_dist['hidden'] = x


        if self.is_discrete:
            latent_dist['discrete'] = [F.softmax(dense_alpha(x), dim=1) for dense_alpha in self.dense_alphas]
        
        return latent_dist

class DecoderVAE_conv(nn.Module):
    def __init__(self, input_dim, latent_dim=10, activation='relu', **kwargs):
        """ 
        Gaussian decoder $p_{\theta}(x|z) $ with convolutional 
        architecture used in [1].
        
        The mean and log-variance of the reconstruction $\hat{x}$ 
        is again parameterized by the decoder.
        
        [1] Locatello et. al., "Challenging Common Assumptions
        in the Unsupervised Learning of Disentangled 
        Representations", arXiv:1811.12359 (2018).
        """
        
        super(DecoderVAE_conv, self).__init__()
        
        self.input_dim = input_dim
        im_channels = self.input_dim[0]
        kernel_dim = 4
        hidden_dim = 256
        hidden_channels = 64
        n_ch1 = 64
        n_ch2 = 32
        cnn_kwargs = dict(stride=2, padding=1)
        # (leaky_relu, relu, elu)
        self.activation = getattr(F, activation)
        self.reshape = (hidden_channels, kernel_dim, kernel_dim)
        
        self.dense1 = nn.Linear(latent_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, np.product((self.reshape)))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            self.upconv_i = nn.ConvTranspose2d(n_ch1, n_ch1, kernel_dim, **cnn_kwargs)
        
        self.upconv1 = nn.ConvTranspose2d(n_ch1, n_ch2, kernel_dim, **cnn_kwargs)
        self.upconv2 = nn.ConvTranspose2d(n_ch2, n_ch2, kernel_dim, **cnn_kwargs)
        self.upconv3 = nn.ConvTranspose2d(n_ch2, im_channels, kernel_dim, **cnn_kwargs)
        
    def forward(self, z):
        
        batch_size = z.size(0)
        x = self.activation(self.dense1(z))
        x = self.activation(self.dense2(x))
        x = x.view((batch_size, *self.reshape))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            x = self.activation(self.upconv_i(x))
            
        x = self.activation(self.upconv1(x))
        x = self.activation(self.upconv2(x))
        logits = self.upconv3(x)
        
        # Implicitly assume output is Bernoulli distributed - bad?
        out = torch.sigmoid(logits)
        
        return out


class MLPEncoder(nn.Module):
    """ For image data.
    """
    def __init__(self, input_dim, latent_spec, **kwargs):
        super(MLPEncoder, self).__init__()

        hidden_dim = 256
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim_continuous = latent_spec['continuous']

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, hidden_dim)

        self.dense_mu = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.dense_logvar = nn.Linear(hidden_dim, self.latent_dim_continuous)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        h = x.view(-1, self.input_dim[1] * self.input_dim[2])
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.hidden_dim)
        latent_dist = {}

        mu = self.dense_mu(z)
        logvar = self.dense_logvar(z)

        latent_dist['continuous'] = [mu, logvar]
        latent_dist['hidden'] = z

        return latent_dist



class MLPDecoder(nn.Module):
    """ For image data.
    """
    def __init__(self, input_dim, latent_dim=10, output_dim=4096, **kwargs):
        super(MLPDecoder, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_dim)

        self.act = nn.ReLU(inplace=True)  # or nn.Tanh

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        # h = self.act(self.fc3(h))
        h = self.fc4(h)

        logits = h.view(z.size(0), self.input_dim[0], self.input_dim[1], self.input_dim[2])
        reconstruction = torch.sigmoid(logits)

        return reconstruction
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from utils import normalization

class Encoder(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=16,
                 channel_norm=True):

        """ 
        Encoder with convolutional architecture proposed in [1].
        Projects image x ([C_in,256,256]) into a feature map of size C x W/16 x H/16
        ========
        Arguments:
        image_dims:  Dimensions of input image, (C_in,H,W)
        batch_size: Number of instances per minibatch
        C:          Bottleneck depth, controls bits-per-pixel
                    C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Encoder, self).__init__()
        
        im_channels = self.image_dims[0]
        kernel_dim = 3
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        self.activation = getattr(F, activation)  # (leaky_relu, relu, elu)
        
        if channel_norm is True:
            interlayer_norm = normalization.ChannelNorm2D_wrap
        else:
            interlayer_norm = normalization.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        heights = (2**i for i in range(4,9))[::-1]
        widths = heights
        H1, H2, H3, H4, H5 = heights
        W1, W2, W3, W4, W5 = widths 

        # (262,262) -> (256,256), with implicit padding
        self.conv1 = nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1)
        self.norm1 = interlayer_norm((batch_size, filters[0], H1, W1), **norm_kwargs)

        # (256,256) -> (128,128)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs)
        self.norm2 = interlayer_norm((batch_size, filters[1], H2, W2), **norm_kwargs)

        # (128,128) -> (64,64)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs)
        self.norm3 = interlayer_norm((batch_size, filters[2], H3, W3), **norm_kwargs)

        # (64,64) -> (32,32)
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs)
        self.norm4 = interlayer_norm((batch_size, filters[3], H4, W4), **norm_kwargs)

        # (32,32) -> (16,16)
        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs)
        self.norm5 = interlayer_norm((batch_size, filters[4], H5, W5), **norm_kwargs)
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_out = nn.Conv2d(filters[4], C, kernel_dim, stride=1, **cnn_kwargs)
        
                
    def forward(self, x):
        
        batch_size = x.size(0)

        x = self.pre_pad(x)
        x = self.activation(self.conv1(x))
        x = self.norm1(x)

        x = self.asymmetric_pad(x)
        x = self.activation(self.conv2(x))
        x = self.norm2(x)

        x = self.asymmetric_pad(x)
        x = self.activation(self.conv3(x))
        x = self.norm3(x)

        x = self.asymmetric_pad(x)
        x = self.activation(self.conv4(x))
        x = self.norm4(x)

        x = self.asymmetric_pad(x)
        x = self.activation(self.conv5(x))
        x = self.norm5(x)
        
        x = self.post_pad(x)
        out = self.conv_out(x)
        
        # Reshape for quantization
        out = out.view((batch_size, -1))
        
        return out

class Discriminator(nn.Module):
    def __init__(self, image_dims, context_dims, C, spectral_norm=True):
        """ 
        Convolutional patchGAN discriminator proposed in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()
        
        self.image_dims = image_dims
        self.context_dims = context_dim
        im_channels = self.image_dims[0]
        kernel_dim = 4
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, 12, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.Identity

        # (256,256) -> (256,256), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels, filters[0], kernel_dim, **cnn_kwargs))

        # (256,256) -> (128,128)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, 1, stride=1)

    def forward(self, x, y):

        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.context_conv(y))
        y = self.context_upsample(y)

        x = torch.cat((x,y), dim=1)
        
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        
        out = torch.sigmoid(self.conv_out(x))
        
        return out
        
"""
TEMPLATES
=====================
Encoders / Decoders
=====================
"""

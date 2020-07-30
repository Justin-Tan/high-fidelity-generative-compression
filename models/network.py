import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

# Custom
from utils import normalization

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = input_dims[1]
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=True)

        if channel_norm is True:
            self.interlayer_norm = normalization.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = normalization.InstanceNorm2D_wrap

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(input_dims, **norm_kwargs)
        self.norm2 = self.interlayer_norm(input_dims, **norm_kwargs)

    def forward(self, x):

        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)


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
        
        kernel_dim = 3
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        im_channels = image_dims[0]
        assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=True)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        
        if channel_norm is True:
            self.interlayer_norm = normalization.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = normalization.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        heights = [2**i for i in range(4,9)][::-1]
        widths = heights
        H1, H2, H3, H4, H5 = heights
        W1, W2, W3, W4, W5 = widths 

        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1),
            self.interlayer_norm((batch_size, filters[0], H1, W1), **norm_kwargs),
            self.activation(),
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[1], H2, W2), **norm_kwargs),
            self.activation(),
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[2], H3, W3), **norm_kwargs),
            self.activation(),
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[3], H4, W4), **norm_kwargs),
            self.activation(),
        )

        # (32,32) -> (16,16)
        self.conv_block5 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[4], H5, W5), **norm_kwargs),
            self.activation(),
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[4], C, kernel_dim, stride=1),
        )
        
                
    def forward(self, x):
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        out = self.conv_block_out(x)
                
        return out


class Generator(nn.Module):
    def __init__(self, input_dims, batch_size, C=16, activation='relu',
                 n_residual_blocks=8, channel_norm=True):

        """ 
        Generator with convolutional architecture proposed in [1].
        Upscales quantized encoder output into feature map of size C x W x H.
        Expects input size (C,16,16)
        ========
        Arguments:
        input_dims: Dimensions of quantized representation, (C,H,W)
        batch_size: Number of instances per minibatch
        C:          Encoder bottleneck depth, controls bits-per-pixel
                    C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Generator, self).__init__()
        
        kernel_dim = 3
        filters = (960, 480, 240, 120, 60)
        self.n_residual_blocks = n_residual_blocks
        assert input_dims == (C, 16, 16), 'Inconsistent input dims to generator'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=True)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        
        if channel_norm is True:
            self.interlayer_norm = normalization.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = normalization.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        H0, W0 = input_dims[1:]
        heights = [2**i for i in range(5,9)]
        widths = heights
        H1, H2, H3, H4 = heights
        W1, W2, W3, W4 = widths 

        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm((batch_size, C, H0, W0), **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(C, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm((batch_size, filters[0], H0, W0), **norm_kwargs),
        )

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(input_dims=(batch_size, filters[0], H0, W0), 
                channel_norm=channel_norm, activation=activation)
            self.add_module('resblock_{}'.format(str(m)), resblock_m)
        
        # TODO Transposed conv. alternative: https://distill.pub/2016/deconv-checkerboard/
        # (16,16) -> (32,32)
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[1], H1, W1), **norm_kwargs),
            self.activation(),
        )

        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[2], H2, W2), **norm_kwargs),
            self.activation(),
        )

        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[3], H3, W3), **norm_kwargs),
            self.activation(),
        )

        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm((batch_size, filters[4], H4, W4), **norm_kwargs),
            self.activation(),
        )

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 3, kernel_size=(7,7), stride=1),
        )


    def forward(self, x):
        
        x = self.conv_block_init(x)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, 'resblock_{}'.format(str(m)))
            x = resblock_m(x)

        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        x = self.upconv_block4(x)
        out = self.conv_block_out(x)

        return torch.tanh(out)


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
                        C = {2,4,8,16}, C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()
        
        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
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
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels + context_C_out, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

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


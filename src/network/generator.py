import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom
from src.normalisation import channel, instance

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = input_dims[1]
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

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

class Generator(nn.Module):
    def __init__(self, input_dims, batch_size, C=16, activation='relu',
                 n_residual_blocks=8, channel_norm=True, sample_noise=False,
                 noise_dim=32):

        """ 
        Generator with convolutional architecture proposed in [1].
        Upscales quantized encoder output into feature map of size C x W x H.
        Expects input size (C,16,16)
        ========
        Arguments:
        input_dims: Dimensions of quantized representation, (C,H,W)
        batch_size: Number of instances per minibatch
        C:          Encoder bottleneck depth, controls bits-per-pixel
                    C = 220 used in [1].

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Generator, self).__init__()
        
        kernel_dim = 3
        filters = [960, 480, 240, 120, 60]
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = 4
        
        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

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
            self.interlayer_norm(C, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(C, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        if sample_noise is True:
            # Concat noise with latent representation
            filters[0] += self.noise_dim

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(input_dims=(batch_size, filters[0], H0, W0), 
                channel_norm=channel_norm, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        # (16,16) -> (32,32)
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 3, kernel_size=(7,7), stride=1),
        )


    def forward(self, x):
        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        x = self.upconv_block4(x)
        out = self.conv_block_out(x)

        return out



if __name__ == "__main__":

    C = 8
    y = torch.randn([3,C,16,16])
    y_dims = y.size()
    G = Generator(y_dims[1:], y_dims[0], C=C, n_residual_blocks=3, sample_noise=True)

    x_hat = G(y)
    print(x_hat.size())
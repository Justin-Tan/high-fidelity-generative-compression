"""
Stitches submodels together.
"""
import numpy as np

from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
import hific.perceptual_similarity as ps
from hific.models import network, hyperprior
from hific.utils import helpers, initialization, datasets, math, distributions

from default_config import model_modes, model_types, args, directories

Nodes = namedtuple(
    "Nodes",                    # Expected ranges for RGB:
    ["input_image",             # [0, 255]
     "input_image_scaled",      # [0, 1]
     "reconstruction",          # [0, 255]
     "reconstruction_scaled",   # [0, 1]
     "latent_quantized"])       # Latent post-quantization.


class HificModel(nn.Module):

    def __init__(self, args, logger, mode=model_modes.TRAINING, model_type=model_types.COMPRESSION):
        super(HificModel, self).__init__()

        """
        Builds hific model from submodels.
        """

        self.args = args
        self.mode = mode
        self.model_type = model_type
        self.logger = logger

        if not isinstance(self.model_type, model_types):
            raise ValueError("Invalid model_type: [{}]".format(self.model_type))

        if self.model_type == model_types.COMPRESSION_GAN:
            assert self.args.discriminator_steps > 0, 'Must specify nonzero training steps for D!'
            self.discriminator_steps = self.args.discriminator_steps
            self.logger.info('Generative mode enabled.')
        else:
            self.discriminator_steps = 0

        self.image_dims = self.args.image_dims  # Assign from dataloader
        self.batch_size = self.args.batch_size

        self.Encoder = network.Encoder(self.image_dims, self.batch_size, C=self.args.latent_channels,
            channel_norm=self.args.use_channel_norm)
        self.Generator = network.Generator(self.image_dims, self.batch_size, C=self.args.latent_channels,
            channel_norm=self.args.use_channel_norm)

        self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.args.latent_channels)

        self.Discriminator = network.Discriminator(image_dims=self.image_dims,
            context_dims=self.args.latent_dims, C=self.args.latent_channels)
        
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net_lin', net='alex', use_gpu=True)

    def compression_forward(self, x):
        """x: Input image, range [0, 255]."""

        if self.mode = model_modes.VALIDATION and (self.training is False):
            image_dims = tuple(torch.size(x)[1:])
            n_downsamples = self.Encoder.n_downsampling_layers
            x = _pad(x, image_dims, n_downsamples)

        x = x.float() / 255.

    def perceptual_loss(self, x_gen, x_real):
        """Assumes inputs are in [0, 1]."""
        return torch.mean(self.perceptual_loss(x_gen, x_real, normalize=True))

if __name__ == '__main__':

    start_time = time.time()
    logger = helpers.logger_setup(logpath=os.path.join(directories.experiments, 'logs'), filepath=os.path.abspath(__file__))

    model = HificModel(args, logger)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(helpers.count_parameters(model)))

    for n, p in model.named_parameters():
        logger.info(n)

    x = torch.randn([10, 2, 256, 256])
    out = model(x)
    print('Out shape', out.size())

    logger.info('Delta t {:.3f}s'.format(time.time() - start_time))
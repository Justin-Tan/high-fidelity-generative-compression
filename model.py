"""
Stitches submodels together.
"""
import numpy as np
import time, os

from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
import hific.perceptual_similarity as ps
from hific.submodels import network, hyperprior
from hific.utils import helpers, datasets, math, losses

from default_config import ModelModes, ModelTypes, hific_args, directories

Intermediates = namedtuple("Intermediates",
    ["input_image",             # [0, 1] (after scaling from [0, 255])
     "reconstruction",          # [0, 1]
     "latents_quantized",        # Latents post-quantization.
     "n_bpp",                   # Differential entropy estimate.
     "q_bpp"])                  # Shannon entropy estimate.

Disc_out= namedtuple("disc_out",
    ["D_real", "D_gen", "D_real_logits", "D_gen_logits"])

class HificModel(nn.Module):

    def __init__(self, args, logger, model_mode=ModelModes.TRAINING, model_type=ModelTypes.COMPRESSION):
        super(HificModel, self).__init__()

        """
        Builds hific model from submodels.
        """
        self.args = args
        self.model_mode = model_mode
        self.model_type = model_type
        self.logger = logger
        self.step_counter = 0

        if not hasattr(ModelTypes, self.model_type.upper()):
            raise ValueError("Invalid model_type: [{}]".format(self.model_type))
        if not hasattr(ModelModes, self.model_mode.upper()):
            raise ValueError("Invalid model_mode: [{}]".format(self.model_mode))

        self.image_dims = self.args.image_dims  # Assign from dataloader
        self.batch_size = self.args.batch_size

        self.Encoder = network.Encoder(self.image_dims, self.batch_size, C=self.args.latent_channels,
            channel_norm=self.args.use_channel_norm)
        self.Generator = network.Generator(self.image_dims, self.batch_size, C=self.args.latent_channels,
            n_residual_blocks=self.args.n_residual_blocks, channel_norm=self.args.use_channel_norm)

        self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.args.latent_channels)

        self.amortization_models = [self.Encoder, self.Generator]
        self.amortization_models.extend(self.Hyperprior.amortization_models)

        # Use discriminator if GAN mode enabled and in training/validation
        self.use_discriminator = (
            self.model_type == ModelTypes.COMPRESSION_GAN
            and (self.model_mode != ModelModes.EVALUATION)
        )

        if self.use_discriminator is True:
            assert self.args.discriminator_steps > 0, 'Must specify nonzero training steps for D!'
            self.discriminator_steps = self.args.discriminator_steps
            self.logger.info('GAN mode enabled. Training discriminator for {} steps.'.format(
                self.discriminator_steps))
            self.Discriminator = network.Discriminator(image_dims=self.image_dims,
                context_dims=self.args.latent_dims, C=self.args.latent_channels)
        else:
            self.discriminator_steps = 0
            self.Discriminator = None

        
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())
        

    def compression_forward(self, x):
        """
        Forward pass through encoder, hyperprior, and decoder.

        Inputs
        x:  Input image. Format (N,C,H,W), range [0, 255].
            torch.Tensor
        
        Outputs
        intermediates: NamedTuple of intermediate values
        """
        image_dims = tuple(x.size()[1:])  # (C,H,W)

        if self.model_mode == ModelModes.VALIDATION and (self.training is False):
            n_downsamples = self.Encoder.n_downsampling_layers
            factor = 2 ** n_downsamples
            logger.info('Padding to {}'.format(factor))
            x = helpers.pad_factor(x, image_dims, factor)

        # Scale range to [0,1]
        x = torch.div(x, 255.)

        # Encoder forward pass
        y = self.Encoder(x)
        hyperinfo = self.Hyperprior(y)

        latents_quantized = hyperinfo.decoded
        total_nbpp = hyperinfo.total_nbpp
        total_qbpp = hyperinfo.total_qbpp

        reconstruction = self.Generator(y)
        # Undo padding
        reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]
        
        intermediates = Intermediates(x, reconstruction, latents_quantized, 
            total_nbpp, total_qbpp)

        return intermediates

    def discriminator_forward(self, intermediates, generator_train):
        """ Train on gen/real batches simultaneously. """
        x_gen = intermediates.reconstruction
        x_real = intermediates.input_image

        # Alternate between training discriminator and compression models
        if generator_train is False:
            x_gen = x_gen.detach()

        D_in = torch.cat([x_real, x_gen], dim=0)

        latents = intermediates.latents_quantized.detach()
        # latents = torch.cat([latents, latents], dim=0)
        latents = torch.repeat_interleave(latents, 2, dim=0)

        D_out, D_out_logits = self.Discriminator(D_in, latents)
        print(D_out.size())

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        # Tensorboard
        # real_response, gen_response = D_real.mean(), D_fake.mean()

        return Disc_out(D_real, D_gen, D_real_logits, D_gen_logits)

    def distortion_loss(self, x_gen, x_real):
        # loss in [0,255] space but normalized by 255 to not be too big
        mse = self.squared_difference(x_gen, x_real) # / 255.
        return torch.mean(mse)

    def perceptual_loss_wrapper(self, x_gen, x_real):
        """ Assumes inputs are in [0, 1]. """
        LPIPS_loss = self.perceptual_loss.forward(x_gen, x_real, normalize=True)
        return torch.mean(LPIPS_loss)

    def compression_loss(self, intermediates):
        
        x_real = intermediates.input_image
        x_gen = intermediates.reconstruction

        weighted_distortion = self.args.k_M * self.distortion_loss(x_gen, x_real)
        weighted_perceptual = self.args.k_P * self.perceptual_loss_wrapper(x_gen, x_real)
        print('Distortion loss size', weighted_distortion.size())
        print('Perceptual loss size', weighted_perceptual.size())

        weighted_rate = losses.weighted_rate_loss(self.args, total_nbpp=intermediates.n_bpp,
            total_qbpp=intermediates.q_bpp, step_counter=self.step_counter)

        print('Weighted rate loss size', weighted_rate.size())
        weighted_R_D_loss = weighted_rate + weighted_distortion
        weighted_compression_loss = weighted_R_D_loss + weighted_perceptual

        print('Weighted R-D loss size', weighted_R_D_loss.size())

        if self.use_discriminator is True:
            disc_out = self.discriminator_forward(intermediates, generator_train=True)
            G_loss = losses.gan_loss(disc_out, mode='generator_loss')
            print('G loss size', G_loss.size())
            weighted_G_loss = self.args.beta * G_loss
            weighted_compression_loss += weighted_G_loss

        return weighted_compression_loss


    def discriminator_loss(self, intermediates):

        disc_out = self.discriminator_forward(intermediates, generator_train=False)
        D_loss = losses.gan_loss(disc_out, mode='discriminator_loss')

        return D_loss

    def forward(self, x):

        self.step_counter += 1
        intermediates = self.compression_forward(x)

        if self.model_mode == ModelModes.EVALUATION:
            reconstruction = torch.mul(intermediates.reconstruction, 255.)
            reconstruction = torch.clamp(reconstruction, min=0., max=255.)
            return reconstruction, intermediates.qbpp

        compression_model_loss = self.compression_loss(intermediates)

        if self.use_discriminator:
            discriminator_model_loss = self.discriminator_loss(intermediates)
            return compression_model_loss, discriminator_model_loss

        return compression_model_loss

if __name__ == '__main__':

    start_time = time.time()
    logger = helpers.logger_setup(logpath=os.path.join(directories.experiments, 'logs'), filepath=os.path.abspath(__file__))
    logger.info('Starting forward pass ...')
    device = helpers.get_device()
    logger.info('Using device {}'.format(device))
    model = HificModel(hific_args, logger, model_type=ModelTypes.COMPRESSION_GAN)
    model.to(device)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(helpers.count_parameters(model)))

    for n, p in model.named_parameters():
        logger.info(n)

    x = torch.randn([10, 3, 256, 256]).to(device)
    compression_loss, disc_loss = model(x)
    print('Compression loss shape', compression_loss.size())
    print('Disc loss shape', disc_loss.size())

    logger.info('Delta t {:.3f}s'.format(time.time() - start_time))


"""
Stitches submodels together.
"""
import numpy as np
import time, os
import itertools

from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from src.network import hyperprior, encoder, generator, discriminator
from src.helpers import maths, datasets, utils
from src.loss import losses
from src.loss.perceptual_similarity import perceptual_loss as ps 

from default_config import ModelModes, ModelTypes, hific_args, directories

Intermediates = namedtuple("Intermediates",
    ["input_image",             # [0, 1] (after scaling from [0, 255])
     "reconstruction",          # [0, 1]
     "latents_quantized",       # Latents post-quantization.
     "n_bpp",                   # Differential entropy estimate.
     "q_bpp"])                  # Shannon entropy estimate.

Disc_out= namedtuple("disc_out",
    ["D_real", "D_gen", "D_real_logits", "D_gen_logits"])

class Model(nn.Module):

    def __init__(self, args, logger, storage_train=defaultdict(list), storage_test=defaultdict(list), model_mode=ModelModes.TRAINING, 
            model_type=ModelTypes.COMPRESSION):
        super(Model, self).__init__()

        """
        Builds hific model from submodels in network.
        """
        self.args = args
        self.model_mode = model_mode
        self.model_type = model_type
        self.logger = logger
        self.log_interval = args.log_interval
        self.storage_train = storage_train
        self.storage_test = storage_test
        self.step_counter = 0

        if not hasattr(ModelTypes, self.model_type.upper()):
            raise ValueError("Invalid model_type: [{}]".format(self.model_type))
        if not hasattr(ModelModes, self.model_mode.upper()):
            raise ValueError("Invalid model_mode: [{}]".format(self.model_mode))

        self.image_dims = self.args.image_dims  # Assign from dataloader
        self.batch_size = self.args.batch_size

        self.Encoder = encoder.Encoder(self.image_dims, self.batch_size, C=self.args.latent_channels,
            channel_norm=self.args.use_channel_norm)
        self.Generator = generator.Generator(self.image_dims, self.batch_size, C=self.args.latent_channels,
            n_residual_blocks=self.args.n_residual_blocks, channel_norm=self.args.use_channel_norm)

        self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.args.latent_channels,
            likelihood_type=args.likelihood_type)

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
            self.Discriminator = discriminator.Discriminator(image_dims=self.image_dims,
                context_dims=self.args.latent_dims, C=self.args.latent_channels)
        else:
            self.discriminator_steps = 0
            self.Discriminator = None

        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=[args.gpu])
        
    def store_loss(self, key, loss):
        assert type(loss) == float, 'Call .item() on loss before storage'

        if self.training is True:
            storage = self.storage_train
        else:
            storage = self.storage_test

        if self.writeout is True:
            storage[key].append(loss)


    def compression_forward(self, x):
        """
        Forward pass through encoder, hyperprior, and decoder.

        Inputs
        x:  Input image. Format (N,C,H,W), range [0,1],
            or [-1,1] if args.normalize_image is True
            torch.Tensor
        
        Outputs
        intermediates: NamedTuple of intermediate values
        """
        image_dims = tuple(x.size()[1:])  # (C,H,W)

        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            n_encoder_downsamples = self.Encoder.n_downsampling_layers
            factor = 2 ** n_encoder_downsamples
            self.logger.info('Padding input image to {}'.format(factor))
            x = utils.pad_factor(x, x.size()[2:], factor)

        # Encoder forward pass
        y = self.Encoder(x)

        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            n_hyperencoder_downsamples = self.Hyperprior.analysis_net.n_downsampling_layers
            factor = 2 ** n_hyperencoder_downsamples
            self.logger.info('Padding latents to {}'.format(factor))
            y = utils.pad_factor(y, y.size()[2:], factor)

        hyperinfo = self.Hyperprior(y, spatial_shape=x.size()[2:])

        latents_quantized = hyperinfo.decoded
        total_nbpp = hyperinfo.total_nbpp
        total_qbpp = hyperinfo.total_qbpp

        # Use quantized latents as input to G
        reconstruction = self.Generator(latents_quantized)
        
        if self.args.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]
        
        intermediates = Intermediates(x, reconstruction, latents_quantized, 
            total_nbpp, total_qbpp)

        return intermediates, hyperinfo

    def discriminator_forward(self, intermediates, train_generator):
        """ Train on gen/real batches simultaneously. """
        x_gen = intermediates.reconstruction
        x_real = intermediates.input_image

        # Alternate between training discriminator and compression models
        if train_generator is False:
            x_gen = x_gen.detach()

        D_in = torch.cat([x_real, x_gen], dim=0)

        latents = intermediates.latents_quantized.detach()
        latents = torch.repeat_interleave(latents, 2, dim=0)

        D_out, D_out_logits = self.Discriminator(D_in, latents)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        return Disc_out(D_real, D_gen, D_real_logits, D_gen_logits)

    def distortion_loss(self, x_gen, x_real):
        # loss in [0,255] space but normalized by 255 to not be too big
        # - Delegate scaling to weighting
        sq_err = self.squared_difference(x_gen*255., x_real*255.) # / 255.
        return torch.mean(sq_err)

    def perceptual_loss_wrapper(self, x_gen, x_real, normalize=True):
        """ Assumes inputs are in [0, 1] if normalize=True, else [-1, 1] """
        LPIPS_loss = self.perceptual_loss.forward(x_gen, x_real, normalize=normalize)
        return torch.mean(LPIPS_loss)

    def compression_loss(self, intermediates, hyperinfo):
        
        x_real = intermediates.input_image
        x_gen = intermediates.reconstruction

        if self.args.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            x_real = (x_real + 1.) / 2.
            x_gen = (x_gen + 1.) / 2.

        distortion_loss = self.distortion_loss(x_gen, x_real)
        perceptual_loss = self.perceptual_loss_wrapper(x_gen, x_real, normalize=True)

        weighted_distortion = self.args.k_M * distortion_loss
        weighted_perceptual = self.args.k_P * perceptual_loss

        weighted_rate, rate_penalty = losses.weighted_rate_loss(self.args, total_nbpp=intermediates.n_bpp,
            total_qbpp=intermediates.q_bpp, step_counter=self.step_counter)

        weighted_R_D_loss = weighted_rate + weighted_distortion
        weighted_compression_loss = weighted_R_D_loss + weighted_perceptual

        # Bookkeeping 
        if (self.step_counter % self.log_interval == 1):
            self.store_loss('rate_penalty', rate_penalty)
            self.store_loss('distortion', distortion_loss.item())
            self.store_loss('perceptual', perceptual_loss.item())
            self.store_loss('n_rate', intermediates.n_bpp.item())
            self.store_loss('q_rate', intermediates.q_bpp.item())
            self.store_loss('n_rate_latent', hyperinfo.latent_nbpp.item())
            self.store_loss('q_rate_latent', hyperinfo.latent_qbpp.item())
            self.store_loss('n_rate_hyperlatent', hyperinfo.hyperlatent_nbpp.item())
            self.store_loss('q_rate_hyperlatent', hyperinfo.hyperlatent_qbpp.item())

            self.store_loss('weighted_rate', weighted_rate.item())
            self.store_loss('weighted_distortion', weighted_distortion.item())
            self.store_loss('weighted_perceptual', weighted_perceptual.item())
            self.store_loss('weighted_R_D', weighted_R_D_loss.item())
            self.store_loss('weighted_compression_loss_sans_G', weighted_compression_loss.item())

        return weighted_compression_loss


    def GAN_loss(self, intermediates, train_generator=False):
        """
        train_generator: Flag to send gradients to generator
        """
        disc_out = self.discriminator_forward(intermediates, train_generator)
        D_loss = losses.gan_loss(disc_out, mode='discriminator_loss')
        G_loss = losses.gan_loss(disc_out, mode='generator_loss')

        # Bookkeeping 
        if (self.step_counter % self.log_interval == 1):
            self.store_loss('D_gen', torch.mean(disc_out.D_gen).item())
            self.store_loss('D_real', torch.mean(disc_out.D_real).item())
            self.store_loss('disc_loss', D_loss.item())
            self.store_loss('gen_loss', G_loss.item())
            self.store_loss('weighted_gen_loss', (self.args.beta * G_loss).item())

        return D_loss, G_loss

    def forward(self, x, train_generator=False, return_intermediates=False, writeout=True):

        self.writeout = writeout

        losses = dict()
        if train_generator is True:
            # Define a 'step' as one cycle of G-D training
            self.step_counter += 1

        intermediates, hyperinfo = self.compression_forward(x)

        if self.model_mode == ModelModes.EVALUATION:

            reconstruction = intermediates.reconstruction
            
            if self.args.normalize_input_image is True:
                # [-1.,1.] -> [0.,1.]
                reconstruction = (reconstruction + 1.) / 2.

            # reconstruction = torch.mul(reconstruction, 255.)
            # reconstruction = torch.clamp(reconstruction, min=0., max=255.)
            reconstruction = torch.clamp(reconstruction, min=0., max=1.)
            return reconstruction, intermediates.q_bpp, intermediates.n_bpp

        compression_model_loss = self.compression_loss(intermediates, hyperinfo)

        if self.use_discriminator is True:
            # Only send gradients to generator when training generator via
            # `train_generator` flag
            D_loss, G_loss = self.GAN_loss(intermediates, train_generator)
            weighted_G_loss = self.args.beta * G_loss
            compression_model_loss += weighted_G_loss
            losses['disc'] = D_loss
        
        losses['compression'] = compression_model_loss

        # Bookkeeping 
        if (self.step_counter % self.log_interval == 1):
            self.store_loss('weighted_compression_loss', compression_model_loss.item())

        if return_intermediates is True:
            return losses, intermediates
        else:
            return losses

if __name__ == '__main__':

    logger = utils.logger_setup(logpath=os.path.join(directories.experiments, 'logs'), filepath=os.path.abspath(__file__))
    device = utils.get_device()
    logger.info(f'Using device {device}')
    storage_train = defaultdict(list)
    storage_test = defaultdict(list)
    model = Model(hific_args, logger, storage_train, storage_test, model_type=ModelTypes.COMPRESSION_GAN)
    model.to(device)

    logger.info(model)

    transform_param_names = list()
    transform_params = list()
    logger.info('ALL PARAMETERS')
    for n, p in model.named_parameters():
        if ('Encoder' in n) or ('Generator' in n):
            transform_param_names.append(n)
            transform_params.append(p)
        if ('analysis' in n) or ('synthesis' in n):
            transform_param_names.append(n)
            transform_params.append(p)      
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    amortization_named_parameters = itertools.chain.from_iterable(
            [am.named_parameters() for am in model.amortization_models])
    for n, p in amortization_named_parameters:
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    for n, p in zip(transform_param_names, transform_params):
        logger.info(f'{n} - {p.shape}')

    logger.info('HYPERPRIOR PARAMETERS')
    for n, p in model.Hyperprior.hyperlatent_likelihood.named_parameters():
        logger.info(f'{n} - {p.shape}')

    logger.info('DISCRIMINATOR PARAMETERS')
    for n, p in model.Discriminator.named_parameters():
        logger.info(f'{n} - {p.shape}')

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size: {} MB".format(utils.count_parameters(model) * 4. / 10**6))

    shape = [10, 3, 256, 256]
    logger.info('Starting forward pass with input shape {}'.format(shape))

    start_time = time.time()
    x = torch.randn(shape).to(device)
    losses = model(x)
    compression_loss, disc_loss = losses['compression'], losses['disc']

    logger.info('Delta t {:.3f}s'.format(time.time() - start_time))



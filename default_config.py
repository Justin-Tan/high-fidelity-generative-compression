#!/usr/bin/env python3

"""
Default arguments from [1]. Entries can be manually overriden via
command line arguments in `train.py`.

[1]: arXiv 2006.09965
"""

class ModelTypes(object):
    COMPRESSION = 'compression'
    COMPRESSION_GAN = 'compression_gan'

class ModelModes(object):
    TRAINING = 'training'
    VALIDATION = 'validation'
    EVALUATION = 'evaluation'  # actual entropy coding

class Datasets(object):
    OPENIMAGES = 'openimages'
    CITYSCAPES = 'cityscapes'
    JETS = 'jetimages'

class DatasetPaths(object):
    OPENIMAGES = 'data/openimages'
    CITYSCAPES = ''
    JETS = ''

class directories(object):
    experiments = 'experiments'

class args(object):
    """
    Shared config
    """
    name = 'hific_v0.1'
    silent = True
    n_epochs = 8
    n_steps = 1e6
    batch_size = 8
    log_interval = 1000
    save_interval = 50000
    gpu = 0
    multigpu = True
    dataset = Datasets.OPENIMAGES
    dataset_path = DatasetPaths.OPENIMAGES
    shuffle = True

    # GAN params
    discriminator_steps = 0
    model_mode = ModelModes.TRAINING
    sample_noise = False
    noise_dim = 32

    # Architecture params - defaults correspond to Table 3a) of [1]
    latent_channels = 220
    n_residual_blocks = 9           # Authors use 9 blocks, performance saturates at 5
    lambda_B = 2**(-4)              # Loose rate
    k_M = 0.075 * 2**(-5)           # Distortion
    k_P = 1.                        # Perceptual loss
    beta = 0.15                     # Generator loss
    use_channel_norm = True
    likelihood_type = 'gaussian'    # Latent likelihood model
    normalize_input_image = False   # Normalize inputs to range [-1,1]
    
    # Shapes
    crop_size = 256
    image_dims = (3,256,256)
    latent_dims = (latent_channels,16,16)
    
    # Optimizer params
    learning_rate = 1e-4
    weight_decay = 1e-6

    # Scheduling
    lambda_schedule = dict(vals=[2., 1.], steps=[50000])
    lr_schedule = dict(vals=[1., 0.1], steps=[500000])
    target_schedule = dict(vals=[0.20/0.14, 1.], steps=[50000])  # Rate allowance
    ignore_schedule = False

    # match target rate to lambda_A coefficient
    regime = 'low'  # -> 0.14
    target_rate_map = dict(low=0.14, med=0.3, high=0.45)
    lambda_A_map = dict(low=2**1, med=2**0, high=2**(-1))
    target_rate = target_rate_map[regime]
    lambda_A = lambda_A_map[regime]

    # DLMM
    use_latent_mixture_model = False
    mixture_components = 4
    latent_channels_DLMM = 64

"""
Specialized configs
"""

class mse_lpips_args(args):
    """
    Config for model trained with distortion and 
    perceptual loss only.
    """
    model_type = ModelTypes.COMPRESSION

class hific_args(args):
    """
    Config for model trained with full generative
    loss terms.
    """
    model_type = ModelTypes.COMPRESSION_GAN
    gan_loss_type = 'non_saturating'  # ('non_saturating', 'least_squares')
    discriminator_steps = 1
    sample_noise = False

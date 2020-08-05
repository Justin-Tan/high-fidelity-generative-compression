#!/usr/bin/env python3

"""
Default arguments from [1]. Entries can be manually overriden via
command line arguments in `train.py`.

[1]: arXiv 2006.09965
"""

class model_types(object):
    COMPRESSION = 'compression'
    COMPRESSION_GAN = 'compression_gan'

class model_modes(object):
    TRAINING = 'training'
    VALIDATION = 'validation'  # Monitoring
    EVALUATION = 'evaluation'

class args(object):
    """
    Common config
    """
    name = 'hific_exp'
    silent = True
    n_epochs = 42  # Paper says 2M training steps
    batch_size = 2048
    multigpu = True
    DATASETS = [...]
    save_interval = 42
    shuffle = True
    discriminator_steps = 0

    # Architecture params - Table 3a) of [1]
    latent_channels = 220
    lambda_B = 2**(-4)          # Loose rate
    k_M = 0.075 * 2**(-5)       # Distortion
    k_P = 1.                    # Perceptual loss
    beta = 0.15                 # Generator loss
    use_channel_norm = True

    # Shapes
    input_dims = (3,256,256)
    latent_dims = (latent_channels,16,16)
    
    # Optimizer params
    learning_rate = 1e-4
    weight_decay = 1e-6

    # Scheduling
    lambda_schedule = dict(vals=[2., 1.], steps=[50000])
    lr_schedule = dict(vals=[1., 0.1], steps=[500000])
    target_schedule = dict(vals=[0.20/0.14, 1.], steps=[50000])

    # match target rate to lambda_A coefficient
    target_rate = 0.14
    perceptual_weight = 1.
    lambda_A_R = dict(0.14=2**1, 0.30=2**0, 0.45==2**(-1))
    lambda_A = lambda_A_R[target_rate]

    # Constrain rate:
    # Loss = C * (1/lambda * R + CD * D) + CP * P
    # where lambda = lambda_a if current_bpp > target_rate
    # lambda_b otherwise.
    C = 0.1 * 2. ** -5  # R-D joint coefficient
    CD = 0.75  # distortion
    CP = None  # Generator loss

    lambda_A = 0.1 * 2. ** -6
    lambda_B = 0.1 * 2. ** 1


class mse_lpips_args(args):
    """
    Config for model trained with distortion and 
    perceptual loss only.
    """
    model_type = model_types.COMPRESSION

class hific_args(args):
    """
    Config for model trained with full generative
    loss terms.
    """
    model_type = model_types.COMPRESSION_GAN
    gan_loss = 'non_saturating'  # ('non_saturating', 'least_squares')
    discriminator_steps = 1
    CP=0.15  # Sweep over 0.1 * 1.5 ** x

class directories(object):
    experiments = 'experiments'

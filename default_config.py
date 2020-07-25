#!/usr/bin/env python3

"""
Default arguments. Entries can be manually overriden by command line arguments in `train.py`.
"""

class args(object):
    silent = True
    n_epochs = 42
    batch_size = 2048
    multigpu = True
    LOSSES = ['VAE', 'IWAE', 'beta_VAE', 'annealed_VAE', 'factor_VAE', 'beta_TCVAE', 'beta_TCVAE_sensitive']
    DATASETS = ['mnist', 'dsprites', 'dsprites_scream', 'custom', 'jets']
    save_interval = 42

    loss_type = 'beta_TCVAE_sensitive'  # 'factor_VAE'
    dataset = 'dsprites'
    sampling_bias = False
    shuffle = True
    distribution = 'bernoulli'
    identifier = loss_type
    
    # Optimizer params
    learning_rate = 5e-4
    weight_decay = 1e-4

    # beta/Annealed VAE params
    beta = 4.0
    C_init = 0
    C_fin = 5
    gamma = 30.
    
    # Factor VAE params
    n_layers_D = 5
    n_units_D = 512
    lr_D = 5e-5
    gamma_fvae = 12.
    
    # BTCVAE / BTCVAE-sensitive params
    alpha_btcvae = 1.
    beta_btcvae = 2.
    gamma_btcvae = 1.
    
    # Sensitive isolation
    supervision = True
    sensitive_latent_idx = [1,2,3,4]
    supervision_lagrange_m = 128
    
    # Normalizing flow params
    use_flow = False
    flow_steps = 32
    flow_hidden_dim = 64
    
    # Misc.
    prior = 'normal'
    x_dist = 'bernoulli'
    latent_dim = 8
    latent_spec = {'continuous': latent_dim}
    mlp = False
    hidden_dim = 128

class directories(object):
    checkpoints = 'checkpoints'
    results = 'results'

if args.loss_type == 'factor_VAE':
    # Double batch size
    args.batch_size *= 2
    args.n_epochs *= 2
    
if args.supervision is True:
    args.identifier += '_supervised'
    
if args.sampling_bias is True:
    args.identifier += '_biased'
    args.shuffle = False
      
args.name = 'disVAE'

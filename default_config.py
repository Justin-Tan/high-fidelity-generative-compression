#!/usr/bin/env python3

"""
Default arguments. Entries can be manually overriden by command line arguments in `train.py`.
[1]: arXiv 2006.09965
"""

class args(object):
    silent = True
    n_epochs = 42  # Paper says 2M training steps
    batch_size = 2048
    multigpu = True
    DATASETS = [...]
    save_interval = 24
    shuffle = True

    # Architecture params - Table 3a) of [1]
    C_y = 220
    lambda_b = 2e-4
    k_M = 0.075 * 2**(-5)
    k_P = 1.
    beta = 0.15
    
    # Optimizer params
    learning_rate = 1e-4
    weight_decay = 1e-6

class directories(object):
    checkpoints = 'checkpoints'
    results = 'results'

args.name = 'hific_exp'

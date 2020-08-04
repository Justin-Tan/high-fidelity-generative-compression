"""
Stitches submodels together.
"""
import numpy as np

from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from hific.models import network, hyperprior
from hific.utils import helpers, initialization, datasets, math, distributions

from default_config import model_mode, model_type

Nodes = namedtuple(
    "Nodes",                    # Expected ranges for RGB:
    ["input_image",             # [0, 255]
     "input_image_scaled",      # [0, 1]
     "reconstruction",          # [0, 255]
     "reconstruction_scaled",   # [0, 1]
     "latent_quantized"])       # Latent post-quantization.

class hific_model(nn.Module):
    """
    Builds hific model from submodels.
    """
    def __init__(self, args, mode=)

def create_model(args, mode=model_mode.TRAINING, model_type=model_type.COMPRESSION):

    self.args = args
    self.mode = mode

    return model



if __name__ == '__main__':

    description = "Density estimation with latent variable models / normalizing flows."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-lpe", "--logs_per_epoch", type=int, default=4, help="Number of times to report metrics per epoch.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=32, help="Number of epochs between checkpointing.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=2048, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument('--loss_type', type=str, default='log-likelihood', help='Label for metadata.')
    general.add_argument('--smoke_test', action='store_true', help='Shut up and train! No extra metrics.')
    general.add_argument(
        '-f', '--flow', type=str, default='no_flow', choices=['no_flow', 'cnf', 'real_nvp', 'maf'],
        help="Type of flow to use to estimate density. No flow defaults to IWELBO baseline.")
    general.add_argument('--eval_end', type=bool, default=True, help='Evaluate metrics at end of training, note: outputs dataframe with scores for each datapoint.')

    # Dataset options
    dataset_args = parser.add_argument_group("Dataset-related options")
    dataset_args.add_argument("-d", "--dataset", type=str, default='jets', help="Training dataset to use.",
        choices=['dsprites', 'custom', 'dsprites_scream', 'jets'], required=True)

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=32, help="Number of passes over training dataset.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=1e-6, help="Coefficient of L2 regularization.")


    cmd_args = parser.parse_args()

    assert (cmd_args.signal_region and cmd_args.sideband_region) is not True, 'Incompatible arguments -sb and -sr.'

    if cmd_args.gpu != 0:
        torch.cuda.set_device(cmd_args.gpu)

    start_time = time.time()
    device = helpers.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    # args_d, cmd_args_d = dictify(args), vars(cmd_args)
    # args_d.update(cmd_args_d)
    # args = helpers.Struct(**args_d)
    args = cmd_args

    args = helpers.setup_generic_signature(args, special_info=args.flow)
    logger = helpers.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('Using GPU ID {}'.format(args.gpu))

    logger.info('Using dataset: {}'.format(args.dataset))
    test_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                train=False,
                                shuffle=True,
                                signal_region=args.signal_region,
                                sideband_region=args.sideband_region,
                                context_dim=args.context_dim)


    train_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                train=True,
                                shuffle=True,
                                signal_region=args.signal_region,
                                sideband_region=args.sideband_region,
                                context_dim=args.context_dim)


    args.n_data = len(train_loader.dataset)
    args.input_dim = train_loader.dataset.input_dim
    args.n_gen_factors = train_loader.dataset.n_gen_factors
    logger.info('Input Dimensions: {}'.format(args.input_dim))
    if args.context_dim > 0:
        assert args.n_gen_factors >= args.context_dim, 'Too many context variables specified! {} vs. {}'.format(args.n_gen_factors, args.context_dim)
        logger.info('Additional contextual dimensions: {}'.format(args.context_dim))

    model = create_model(args)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(helpers.count_parameters(model)))

    n_gpus = torch.cuda.device_count()
    #if 'cnf' not in args.flow:
    #    helpers.summary(model, input_size=[[args.input_dim]] if args.dataset in ['custom','jets'] else args.input_dim, device='cpu')

    if n_gpus > 1 and args.multigpu is True:
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    model = model.to(device)
    parameters = model.parameters()

    logger.info('Optimizing over:')
    encoder_params, decoder_params = list(), list()
    for name, param in model.named_parameters():
        logger.info(name)
        if 'encoder' in name:
            logger.info('Adding {} to enc params'.format(name))
            encoder_params.append(param)
        if 'decoder' in name:
            logger.info('Adding {} to dec params'.format(name))
            decoder_params.append(param)
    param_groups = {'encoder': encoder_params, 'decoder': decoder_params}

    encoder_optimizer = None
    if (args.vae_model == 'sumo') and (args.sumo_reduce_variance is True):
        logger.info('Optimizing encoder to reduce estimator variance.')
        parameters = decoder_params
        encoder_optimizer = torch.optim.Adam(encoder_params, lr=2e-6, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adamax(parameters, lr=args.learning_rate, eps=1.e-7)
    # optimizer = torch.optim.Adadelta(parameters, eps=1.e-7)

    metadata = dict(input_dim=args.input_dim, flow_type=args.flow)
    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    metadata.update(args_d)
    logger.info(metadata)


    """
    Train
    """
    storage = defaultdict(list)
    storage_test = defaultdict(list)
    model, ckpt_path = train(args, model, train_loader, test_loader, device, optimizer, storage, storage_test, logger, 
        log_interval_p_epoch=args.logs_per_epoch, param_groups=param_groups, encoder_optimizer=encoder_optimizer)
    args.ckpt = ckpt_path

    """
    Generate metrics
    """
    if args.eval_end is True:
        evaluate_end(args, model, logger, device)

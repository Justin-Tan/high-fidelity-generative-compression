"""
Learnable generative compression model [1] implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
    arXiv:2006.09965 (2020).
"""
import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from hific.utils import helpers, initialization, datasets, math
from hific.default_config import args, hific_args, mse_lpips_args, directories

# go fast boi!!
# Optimizes cuda kernels by benchmarking
# No dynamic input sizes or u will have a BAD time
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger):

    start_time = time.time()
    model = HificModel(args, logger, model_type=args.model_type)
    logger.info(model)

    logger.info('Trainable parameters:')
    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(helpers.count_parameters(model)))
    logger.info("Estimated size (assuming fp32): {:.3f} MB".format(helpers.count_parameters(model) * 4. / 10**6))

    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model


def test(epoch, counter, data, device, model, epoch_test_loss, storage, best_test_loss, 
         start_time, epoch_start_time, log_interval_p_epoch, logger):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)
        gen_factors = gen_factors.to(device)
        context = gen_factors[:,1:]  # additional context

        test_loss, test_log_px = compute_loss(data, model, context)
        storage['log_prob_per_dim'].append(-test_loss.item()/data.shape[1])
            
        epoch_test_loss += test_loss.item()
        mean_test_loss = epoch_test_loss / counter
        
        best_test_loss = helpers.log_flow(storage, epoch, counter, mean_test_loss, test_loss, 
                                         best_test_loss, start_time, epoch_start_time, 
                                         batch_size=data.shape[0], header='[TEST]', 
                                         log_interval=log_interval_p_epoch, logger=logger)
        
    return best_test_loss, epoch_test_loss



def train(args, model, train_loader, test_loader, device, storage, storage_test, logger):
    
    logger.info('Using device {}'.format(device))
    assert args.log_interval_p_epoch >= 2, 'Show logs more!'
    log_interval = args.n_data / args.batch_size // args.log_interval_p_epoch
    assert log_interval > 1, 'Need more time between logs!'
    
    test_loader_iter = iter(test_loader)
    model.train()
    start_time = time.time()
    
    amortization_parameters = itertools.chain_from_iterable(
        [am.parameters() for am in model.amortization_models]])

    hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

    amortization_opt = torch.optim.Adam(amortization_parameters,
        lr=args.learning_rate)
    hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, 
        lr=args.learning_rate)

    logger.info('Optimizing over:')
    amortization_params, hyperlatent_likelihood_params = list(), list()
    for name, param in model.named_parameters():
        logger.info(name)
        if 'hyperlatent_likelihood' in name:
            logger.info('Adding {} to hyperlatent likelihood params'.format(name))
            hyperlatent_likelihood_params.append(param)
        if ('hyperlatent_likelihood' not in name) and ('Discriminator' not in name):
            logger.info('Adding {} to amortization params'.format(name))
            amortization_params.append(param)

    param_groups = {'amortization': amortization_params, 'hyperlatent_likelihood': hyperlatent_likelihood_params}

    amort_scheduler = torch.optim.lr_scheduler.MultiStepLR(amortization_opt, milestones=[1], gamma=0.1, verbose=True)
    hpl_scheduler = torch.optim.lr_scheduler.MultiStepLR(hyperlatent_likelihood_opt, milestones=[1], gamma=0.1, verbose=True)
    
    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss = []
        epoch_test_loss = 0.
        counter = 0
        epoch_start_time = time.time()
        
        if epoch % args.save_interval == 0 and epoch > 1 and epoch != args.n_epochs:
            ckpt_path = helpers.save_model(model, optimizer, mean_epoch_loss, args.checkpoints_save, epoch, device, args=args)
        
        for idx, (data, gen_factors) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)
            gen_factors = gen_factors.to(device)  
            context = gen_factors[:,1:]  # additional context
            
            try:
                loss, log_px = compute_loss(data, model, context)
                loss.backward()

                #DEC_CLIP_NORM = 60
                #torch.nn.utils.clip_grad_norm_(param_groups['decoder'], DEC_CLIP_NORM)

                optimizer.step()
                optimizer.zero_grad()

                # Train encoder to reduce SUMO variance
                if args.vae_model == 'sumo' and (args.sumo_reduce_variance is True):
                    _, log_px = compute_loss(data, model, context)
                    encoder_loss = torch.mean(torch.square(log_px))
                    encoder_loss.backward()

                    #ENC_CLIP_NORM = 1000
                    #torch.nn.utils.clip_grad_norm_(param_groups['encoder'], ENC_CLIP_NORM)

                    encoder_optimizer.step()
                    encoder_optimizer.zero_grad()

            except KeyboardInterrupt:
                if epoch > 4:
                    logger.warning('Exiting, saving and evaluating on test set.')
                    ckpt_path = helpers.save_model(model, optimizer, mean_epoch_loss, args.checkpoints_save, epoch, device, args=args)
                    return ckpt_path
                else:
                    return

            if idx % log_interval == 1:
                counter += 1
                epoch_loss.append(loss.item())
                mean_epoch_loss = np.mean(epoch_loss)
                storage['log_prob_per_dim'].append(-loss.item()/data.shape[1])
                best_loss = helpers.log_flow(storage, epoch, counter, mean_epoch_loss, loss.item(),
                                best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                log_interval=log_interval_p_epoch, logger=logger)
                try:
                    test_data, test_gen_factors = test_loader_iter.next()
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data, test_gen_factors = test_loader_iter.next()
                test_context = test_gen_factors[:,1:]

                best_test_loss, epoch_test_loss = test(epoch, counter, test_data, test_gen_factors,
                                                       device, model, epoch_test_loss, 
                                                       storage_test, best_test_loss, start_time, epoch_start_time,
                                                       log_interval_p_epoch, logger)

                if args.smoke_test:
                   return None 

                with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Visualization

                model.train()

        mean_epoch_loss = np.mean(epoch_loss)
        mean_test_loss = epoch_test_loss / counter
        scheduler.step(mean_test_loss)
        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch, 
            mean_epoch_loss, mean_test_loss))    
    
    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ckpt_path = helpers.save_model(model, optimizer, mean_epoch_loss, args.checkpoints_save, epoch, device, args=args)
    args.ckpt = ckpt_path
    logger.info("Training complete. Mean time / epoch: {:.3f} s".format((time.time()-start_time)/args.n_epochs))
    s
    return model, ckpt_path


if __name__ == '__main__':

    description = "Learnable generative compression."
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

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=32, help="Number of passes over training dataset.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=1e-6, help="Coefficient of L2 regularization.")

    cmd_args = parser.parse_args()

    if cmd_args.gpu != 0:
        torch.cuda.set_device(cmd_args.gpu)

    start_time = time.time()
    device = helpers.get_device()
    logger.info('Using device {}'.format(device))

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = helpers.Struct(**args_d)
    args = cmd_args

    args = helpers.setup_generic_signature(args, special_info=args.model_type)
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
    args.image_dims = train_loader.dataset.image_dims
    logger.info('Input Dimensions: {}'.format(args.image_dims))

    model = create_model(args, device, logger)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    model = model.to(device)

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    storage = defaultdict(list)
    storage_test = defaultdict(list)
    model, ckpt_path = train(args, model, train_loader, test_loader, device, storage, storage_test, logger)

    """
    TODO
    Generate metrics
    """

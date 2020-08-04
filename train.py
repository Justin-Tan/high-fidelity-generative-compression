"""
Density estimation using discrete flow models/LVMs in Pytorch
Example usage:
python3 density_est.py -f no_flow -d jets -sb -vm iwae -n sb_iwae_32 -nis 32 -z 8 -epochs 1
"""
import numpy as np
import pandas as pd
import os, glob, time, datetime
import logging, pickle, argparse

from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from hific.default_config import directories
from hific.models import losses, network, flows, iw_vae
from hific.utils import helpers, initialization, datasets, evaluate, math, distributions, vis

# go fast boi!!
# Optimizes cuda kernels NO dynamic input sizes or u will have a BAD time
torch.backends.cudnn.benchmark = True

def create_model(args):
    if args.flow == 'no_flow':
        if args.vae_model == 'vae':
            model = iw_vae.VAE(input_dim=args.input_dim,
                               hidden_dim=args.vae_hidden_dim,
                               latent_dim=args.latent_dim)
        elif args.vae_model == 'iwae':
            if args.use_studentT is True:
                iwae = iw_vae.IWAE_st
            else:
                iwae = iw_vae.IWAE
            model = iwae(input_dim=args.input_dim,
                                hidden_dim=args.vae_hidden_dim,
                                latent_dim=args.latent_dim,
                                num_i_samples=args.num_i_samples)
        elif args.vae_model == 'sumo':
            model = iw_vae.SUMO(input_dim=args.input_dim,
                                hidden_dim=args.vae_hidden_dim,
                                latent_dim=args.latent_dim,
                                min_terms=args.min_RR_terms)
    else:
        if args.flow == 'real_nvp':
            subflow = distributions.InvertibleAffineFlow
        elif args.flow == 'maf':
            subflow = distributions.MAF
        
        model = flows.DiscreteFlowModel(input_dim=args.input_dim,
                                        hidden_dim=args.discrete_flow_hidden_dim,
                                        n_flows=args.flow_steps,
                                        base_dist=distributions.StudentT(args.dof),#Normal(),
                                        flow=subflow,
                                        context_dim=args.context_dim)
    
    return model

def evaluate_end(args, model, logger, device):

    def _postprocess(df_sr, df_sb):
        print('Columns', df_sr.columns)
        auxillary = ['label'] + [col for col in df_sr.columns if col.startswith('_')]
        R = np.exp(df_sr.LL - df_sb.LL)
        print('R max', R.max())
        print('R min', R.min())
        df_agg = pd.concat([df_sr[auxillary], R, df_sr.LL, df_sb.LL], axis=1)
        df_agg.columns=auxillary+['R','log_px_data','log_px_bkg']
        df_agg.columns = [c.replace('_','-') for c in df_agg.columns]

        return df_agg

    # Evaluation on independent test set
    test_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=512,
                                logger=logger,
                                train=False,
                                evaluate=True,
                                sampling_bias=False,
                                shuffle=False,
                                signal_region=True,
                                sideband_region=False,
                                context_dim=args.context_dim)

    if args.flow == 'no_flow':
        args.model = 'vae'
    else:
        args.model = 'discrete'

    metrics, df = metric_LL(args, model, logger, test_loader, device)

    return metrics, df

def compute_loss_flow(x, model, context=None):
    return model.log_density(x, context)

def compute_loss_vae(x, model, context=None):
    x_stats, latent_sample, latent_stats = model(x)
    return model.log_px_estimate(x, x_stats, latent_sample, latent_stats)

def compute_loss(x, model, context=None):
    
    if args.flow == 'no_flow':
        log_px = compute_loss_vae(x, model, context)
    else:
        log_px = compute_loss_flow(x, model, context)

    loss = -torch.mean(log_px)
    return loss, log_px

def test(epoch, counter, data, gen_factors, device, model, epoch_test_loss, storage, best_test_loss, 
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



def train(args, model, train_loader, test_loader, device, optimizer, storage, storage_test, logger, 
          log_interval_p_epoch=4, param_groups={}, encoder_optimizer=None):
    
    logger.info('Using device {}'.format(device))
    assert log_interval_p_epoch >= 2, 'Show logs more!'
    log_interval = args.n_data / args.batch_size // log_interval_p_epoch
    assert log_interval > 1, 'Need more time between logs!'
    
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    test_loader_iter = iter(test_loader)
    model.train()
    start_time = time.time()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, 
                                                           factor=0.5, verbose=True)
    
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
                # Transform base distribution to x by running model forward
                if args.flow != 'no_flow' and args.flow != 'maf':
                    with torch.no_grad():
                        x_K, log_px_K = model.sample(test_data.cuda(), test_context.cuda())
                        val_x = test_data.cpu().numpy()
                        val_sample = torch.clamp(x_K, -5,5).cpu().numpy()
                        for i in range(train_loader.dataset.input_dim):
                            vis.compare_histograms_overlay(epoch=epoch, itr=idx, data_gen=val_sample[:,i],
                                data_real=val_x[:,i], save_dir=args.figures_save, name='discreteflow_{}'.format(i))

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
    

    return model, ckpt_path


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


    # VAE options
    vae_args = parser.add_argument_group("VAE-related options")
    vae_args.add_argument("-z", "--latent_dim", type=int, default=4, help="Dimension of latent space.")
    vae_args.add_argument("-vm", "--vae_model", type=str, choices=['vae','iwae','sumo'], default='iwae', help="Type of VAE model,")
    vae_args.add_argument("-nh", "--vae_hidden_dim", type=int, default=128, help="Dimension of latent space.")
    vae_args.add_argument("-nis", "--num_i_samples", type=int, default=32, help="Number of importance samples. Only has effect for IWAE model.")
    vae_args.add_argument("-mrrt", "--min_RR_terms", type=int, default=16, help="Min. number of terms in RR estimator for SUMO.")
    vae_args.add_argument("-sumo_rv", "--sumo_reduce_variance", action="store_true", help="Optimize encoder to reduce variance for SUMO estimator.")
    vae_args.add_argument("-st", "--use_studentT", action="store_true", help="Use student-T distribution for IWAE.")

    # Dataset options
    dataset_args = parser.add_argument_group("Dataset-related options")
    dataset_args.add_argument("-d", "--dataset", type=str, default='jets', help="Training dataset to use.",
        choices=['dsprites', 'custom', 'dsprites_scream', 'jets'], required=True)
    dataset_args.add_argument("-sr", "--signal_region", action='store_true', help="Consider only events in SR. Incompatible w/ -sb")
    dataset_args.add_argument("-sb", "--sideband_region", action='store_true', help="Consider only events in SB. Incompatible w/ -sr")
    dataset_args.add_argument("-c_dim", "--context_dim", type=int, default=0, help="Dimension of contextual information.")

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=32, help="Number of passes over training dataset.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=1e-6, help="Coefficient of L2 regularization.")

    # Discrete flow options
    discrete_flow_args = parser.add_argument_group("Discrete flow - related options")
    discrete_flow_args.add_argument("-flow_steps", "--flow_steps", type=int, default=8, help="Number of transformations in discrete flow.")
    discrete_flow_args.add_argument("-dof", "--dof", type=int, default=1, help="Degrees of freedom for Student T base dist.")
    discrete_flow_args.add_argument("-discrete_flow_hidden_dim", "--discrete_flow_hidden_dim", type=int, default=64,
        help="Hidden dimension for networks defining discrete flow transformations")

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

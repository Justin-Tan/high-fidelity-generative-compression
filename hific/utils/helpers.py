import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import json
import os, time, datetime
import logging

from scipy.stats import entropy
from collections import OrderedDict
from sklearn.metrics import mutual_info_score

META_FILENAME = "specs.json"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")

def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def quick_restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_factor(input_image, spatial_dims, factor):
    """Pad `input_image` (N,C,H,W) such that H and W are divisible by `factor`."""
    H, W = spatial_dims[0], spatial_dims[1]
    pad_H = (factor - (H % factor)) % factor
    pad_W = (factor - (W % factor)) % factor
    return F.pad(input_image, pad=(0, pad_H, 0, pad_W), mode='reflection')

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def get_scheduled_params(param, param_schedule, step_counter):
    # e.g. schedule = dict(vals=[1., 0.1], steps=[N])
    # reduces param value by a factor of 0.1 after N steps
    vals, steps = param_schedule['vals'], param_schedule['steps']
    assert(len(vals) == len(steps)+1), 'Mispecified schedule! - {}'.format(param_schedule)
    idx = np.where(step_counter < np.array(steps + [step_counter+1]))[0][0]
    param *= vals[idx]

    return param

def setup_generic_signature(args, special_info):

    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    if args.name is not None:
        args.name = '{}_{}_{}_{}'.format(args.name, args.dataset, special_info, time_signature)
    else:
        args.name = '{}_{}_{}'.format(args.dataset, special_info, time_signature)

    print(args.name)
    args.snapshot = os.path.join('experiments', args.name)
    args.checkpoints_save = os.path.join(args.snapshot, 'checkpoints')
    args.figures_save = os.path.join(args.snapshot, 'figures')
    args.storage_save = os.path.join(args.snapshot, 'storage')
    makedirs(args.snapshot)
    makedirs(args.checkpoints_save)
    makedirs(args.figures_save)
    makedirs(args.storage_save)

    return args

def save_metadata(metadata, directory='results', filename=META_FILENAME, **kwargs):
    """ Save the metadata of a training directory.
    Parameters
    ----------
    metadata:
        Object to save
    directory: string
        Path to folder where to save model. For example './experiments/mnist'.
    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)  #, **kwargs)

def save_model(model, optimizer, mean_loss, directory, epoch, device, args,
               multigpu=False, second_model=None, second_optimizer=None):
 
    makedirs(directory)
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    metadata = dict(input_dim=args.input_dim, latent_dim=args.latent_dim,
                    model_loss=args.loss_type)

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('_') or 'logger' in n))
    metadata.update(args_d)
    args_d['timestamp'] = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now())
    
    model_name = args.name
    metadata_path = os.path.join(directory, 'metadata/model_{}_metadata_{:%Y_%m_%d_%H:%M}.json'.format(model_name, datetime.datetime.now()))
    makedirs(os.path.join(directory, 'metadata'))
    
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
            
    model_path = os.path.join(directory, '{}_epoch_{}_{:%Y_%m_%d_%H:%M}.pt'.format(model_name, epoch, datetime.datetime.now()))

    if os.path.exists(model_path):
        model_path = os.path.join(directory, '{}_epoch_{}_{:%Y_%m_%d_%H:%M:%S}.pt'.format(model_name, epoch, datetime.datetime.now()))

    save_dict = {   'model_state_dict': model.module.state_dict() if args.multigpu is True else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'mean_epoch_loss': mean_loss,
                    'args': args_d,
                }

    if (second_model is not None) and (second_optimizer is not None):
        save_dict.update({'second_model_state_dict': second_model.module.state_dict() if args.multigpu is True else second_model.state_dict(),
                          'second_optimizer_state_dict': second_optimizer.state_dict()})

    torch.save(save_dict, f=model_path)
    print('Saved model at Epoch {} to {}'.format(epoch, model_path))
    
    model.to(device)  # Move back to device

    print('Model saved to path {}'.format(model_path))
    return model_path
   

def save_model_online(model, optimizer, epoch, save_dir, name):
    save_path = os.path.join(save_dir, '{}_epoch{}.pt'.format(name, epoch))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    print('Model saved to path {}'.format(save_path))
    
def load_model(save_path, device, logger, current_args_d=None, optimizer=None, prediction=True, partial=False):

    checkpoint = torch.load(save_path)
    loaded_args_d = checkpoint['args']
    vae_args_keys = ['dataset', 'loss_type', 'latent_dim', 'supervision', 'supervision_lagrange_m', 'sensitive_latent_idx', 
            'beta', 'gamma', 'gamma_fvae', 'alpha_btcvae', 'beta_btcvae', 'gamma_btcvae']
    cnf_args_keys = ['dims', 'num_blocks', 'time_length', 'train_T', 'divergence_fn', 'nonlinearity', 'rank', 'solver', 
            'atol', 'rtol', 'step_size', 'layer_type', 'test_solver', 'test_atol', 'test_rtol', 'residual', 'rademacher', 
            'batch_norm', 'bn_lag']

    args = Struct(**loaded_args_d)

    if current_args_d is not None:
        for k,v in current_args_d.items():
            try:
                loaded_v = loaded_args_d[k]
            except KeyError:
                logger.warning('Argument {} (value {}) not present in recorded arguments. Overriding with current.'.format(k,v))
                continue

            if loaded_args_d[k] !=v:
                logger.warning('Current argument {} (value {}) does not match recorded argument (value {}). May be overriden using recorded.'.format(k, v, loaded_args_d[k]))

        loaded_vae_args_d = {k: loaded_args_d[k] for k in vae_args_keys}
        current_args_d.update(loaded_vae_args_d)  # Override current VAE-model related args with saved args

        try:
            if current_args_d['flow'] != 'cnf_freeze_vae':
                loaded_cnf_args_d = {k: loaded_args_d[k] for k in cnf_args_keys}
                current_args_d.update(loaded_cnf_args_d) # Override current CNF-model related args with saved args
        except KeyError:
            pass

        # HACK
        current_args_d.update(loaded_args_d)

        args = Struct(**current_args_d)

    try:
        if args.flow == 'no_flow':
            model = vae.VAE(args)
        elif args.flow == 'real_nvp':
            model = vae.realNVP_VAE(args)
        elif args.flow == 'cnf' or args.flow == 'cnf_freeze_vae':
            model = vae.VAE_ODE(args)
        elif args.flow == 'cnf_amort':
            model = vae.VAE_ODE_amortized(args)
            
    except AttributeError:
        model = vae.VAE(args)

    model.load_state_dict(checkpoint['model_state_dict'], strict=not partial)

    if prediction:
        model.eval()
    else:
        model.train()
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return args, model.to(device), optimizer

    return args, model # model.to(device)



def logger_setup(logpath, filepath, package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(logpath, mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def log(storage, epoch, counter, mean_epoch_loss, total_loss, best_loss, start_time, epoch_start_time, 
        batch_size, header='[TRAIN]', log_interval=100, logger=None):
    
    improved = ''
    t0 = epoch_start_time
    
    if total_loss < best_loss:
        best_loss = total_loss
        improved = '[*]'  
    
    storage['epoch'].append(epoch)
    storage['mean_epoch_loss'].append(mean_epoch_loss)
    storage['time'].append(time.time())

    try:
        reconstruction_loss = storage['reconstruction_loss'][-1]
        kl_loss = storage['kl_loss'][-1]
        elbo = storage['ELBO'][-1]
    except IndexError:
        reconstruction_loss, kl_loss, elbo = np.nan, np.nan, np.nan

    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print

    report_f(header)

    if header == '[TRAIN]':
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | ELBO: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, elbo, 
                 reconstruction_loss, kl_loss, int(batch_size*counter*log_interval / ((time.time()-t0))), time.time()-start_time, improved))
    else:
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | ELBO: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
                 "Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, elbo, reconstruction_loss,
                 kl_loss, time.time()-start_time, improved))

    return best_loss

def log_flow(storage, epoch, counter, mean_epoch_loss, total_loss, best_loss, start_time, epoch_start_time, 
        batch_size, header='[TRAIN]', log_interval=100, logger=None):
    
    improved = ''
    t0 = epoch_start_time
    
    if total_loss < best_loss:
        best_loss = total_loss
        improved = '[*]'  
    
    storage['epoch'].append(epoch)
    storage['mean_epoch_loss'].append(mean_epoch_loss)
    storage['time'].append(time.time())

    try:
        log_prob = storage['log_prob_per_dim'][-1]
    except IndexError:
        log_prob = np.nan

    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print

    report_f(header)

    if header == '[TRAIN]':
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | LL/D: {:.3f} | "
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, log_prob, 
                 int(batch_size*counter*log_interval / ((time.time()-t0))), time.time()-start_time, improved))
    else:
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | LL/D: {:.3f} | "
                 "Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, log_prob,
                 time.time()-start_time, improved))

    return best_loss

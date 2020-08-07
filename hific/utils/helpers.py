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

META_FILENAME = "metadata.json"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")

def get_model_device(model):
    """Return the device where the model sits."""
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
        Path to folder where to save model. For example './experiments/runX'.
    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)  #, **kwargs)

def save_model(model, optimizers, mean_epoch_loss, epoch, device, args, multigpu=False):

    directory = args.checkpoints_save
    makedirs(directory)
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    metadata = dict(image_dims=args.image_dims, epoch=epoch, steps=model.step_counter)
    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('_') or 'logger' in n))
    metadata.update(args_d)
    timestamp = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now())
    args_d['timestamp'] = timestamp
    
    model_name = args.name
    metadata_path = os.path.join(directory, 'metadata/model_{}_metadata_{}.json'.format(model_name, timestamp))
    makedirs(os.path.join(directory, 'metadata'))
    
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
            
    model_path = os.path.join(directory, '{}_epoch_{}_{}_{}.pt'.format(model_name, epoch, timestamp))

    if os.path.exists(model_path):
        model_path = os.path.join(directory, '{}_epoch_{}_{:%Y_%m_%d_%H:%M:%S}.pt'.format(model_name, epoch, datetime.datetime.now()))

    save_dict = {   'model_state_dict': model.module.state_dict() if args.multigpu is True else model.state_dict(),
                    'compression_optimizer_state_dict': optimizer['amort'].state_dict(),
                    'hyperprior_optimizer_state_dict': optimizer['hyper'].state_dict(),
                    'epoch': epoch,
                    'steps': model.step_counter,
                    'args': args_d,
                }

    if model.use_discriminator is True:
        save_dict['discriminator_state_dict'] = model.module.Discriminator.state_dict() \
            if args.multigpu is True else model.Discriminator.state_dict()
        save_dict['discriminator_optimizer_state_dict'] = optimizer['disc'].state_dict()

    torch.save(save_dict, f=model_path)
    print('Saved model at Epoch {}, step {} to {}'.format(epoch, model.step_counter, model_path))
    
    model.to(device)  # Move back to device
    return model_path
   

def load_model(save_path, model_class, device, logger, current_args_d=None, optimizers=None, prediction=True):

    from hific.model import HificModel
    checkpoint = torch.load(save_path)
    loaded_args_d = checkpoint['args']

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

        # HACK
        current_args_d.update(loaded_args_d)
        args = Struct(**current_args_d)

    model = HificModel(args, logger, model_type=args.model_type)
    model.load_state_dict(checkpoint['model_state_dict'])

    if prediction:
        model.eval()
    else:
        model.train()
        
    if optimizers is not None:
        optimizers['amort'].load_state_dict(checkpoint['compression_optimizer_state_dict'])
        optimizers['hyper'].load_state_dict(checkpoint['hyperprior_optimizer_state_dict'])
        if model.use_discriminator is True:
            optimizers['disc'].load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        return args, model, optimizers

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


def log(storage, epoch, idx, counter, mean_epoch_loss, current_loss, best_loss, start_time, epoch_start_time, 
        batch_size, header='[TRAIN]', logger=None, **kwargs):
    
    improved = ''
    t0 = epoch_start_time
    
    if current_loss < best_loss:
        best_loss = current_loss
        improved = '[*]'  
    
    storage['epoch'].append(epoch)
    storage['mean_compression_loss'].append(mean_epoch_loss)
    storage['time'].append(time.time())

    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print

    report_f(header)
    report_f('=====')
    if header == '[TRAIN]':
        report_f("Epoch {} | Mean epoch comp. loss: {:.3f} | Current comp. loss: {:.3f} | "
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, current_loss,
                 int(batch_size*idx / ((time.time()-t0))), time.time()-start_time, improved))
        report_f("Rate-Distortion:")
        report_f("Weighted R-D: {:3f} | Weighted Rate: {:.3f} | Weighted Distortion: {:.3f} | Weighted Perceptual: {:.3f} | "
                 "n_bpp: {:.3f} | q_bpp: {:.3f} | Distortion: {:.3f} | Rate Penalty: {:.3f}".format(storage['weighted_R_D'],
                 storage['weighted_rate'], storage['weighted_distortion'], storage['weighted_perceptual'], storage['n_rate'],
                 storage['q_rate'], storage['distortion'], storage['rate_penalty']))
        if model.use_discriminator is True:
            report_f("Generator-Discriminator:")
            report_f("G Loss: {:3f} | D Loss: {:.3f} | D(gen): {:.3f} | D(real): {:.3f}".format(storage['gen_loss'],
                    storage['disc_loss'], storage['D_gen'], storage['D_real']))

    return best_loss


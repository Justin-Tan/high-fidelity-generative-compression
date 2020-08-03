""" Authors: @YannDubs 2019
             @sksq96   2019 """

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
import os, time, datetime
import logging

from scipy.stats import entropy
from collections import OrderedDict
from sklearn.metrics import mutual_info_score

from models import network
from utils import distributions

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

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

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


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                summary[m_key]["output_shape"] = []
                getsize = lambda x: [-1] + [int(np.squeeze(list(o.size())[1:])) for o in x]
                # output.pop('hidden')
                output = {}
                for k, v in zip(output.keys(), output.values()):
                    if isinstance(out, list):
                        output_i = {k: torch.cat(v, axis=-1)}
                    elif isinstance(out, torch.Tensor):
                        output_i = {k: v}
                    output.update(output_i)

                summary[m_key]["output_shape"] += [getsize(output.values())]

                # summary[m_key]["output_shape"] = [
                #    [-1] + list(v.size())[1:] for v in output.values()]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    flatten = lambda x: [elem for sl in x for elem in sl] 

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        os = summary[layer]["output_shape"]
        flat1 = [i for i in os if not isinstance(i, list)]
        flat2 = flatten([i for i in os if isinstance(i, list)])
        os = flat1 + flat2
        total_output += np.prod(os)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    try:
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    except AttributeError:
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

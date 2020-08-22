import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.helpers.utils import get_scheduled_params

def weighted_rate_loss(config, total_nbpp, total_qbpp, step_counter, ignore_schedule=False):
    """
    Heavily penalize the rate with weight lambda_A >> lambda_B if it exceeds 
    some target r_t, otherwise penalize with lambda_B
    """
    lambda_A = get_scheduled_params(config.lambda_A, config.lambda_schedule, step_counter, ignore_schedule)
    lambda_B = get_scheduled_params(config.lambda_B, config.lambda_schedule, step_counter, ignore_schedule)

    assert lambda_A > lambda_B, "Expected lambda_A > lambda_B, got (A) {} <= (B) {}".format(
        lambda_A, lambda_B)

    target_bpp = get_scheduled_params(config.target_rate, config.target_schedule, step_counter, ignore_schedule)

    total_qbpp = total_qbpp.item()
    if total_qbpp > target_bpp:
        rate_penalty = lambda_A
    else:
        rate_penalty = lambda_B
    weighted_rate = rate_penalty * total_nbpp

    return weighted_rate, float(rate_penalty)

def _non_saturating_loss(D_real_logits, D_gen_logits, D_real=None, D_gen=None):

    D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
        target=torch.ones_like(D_real_logits))
    D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.zeros_like(D_gen_logits))
    D_loss = D_loss_real + D_loss_gen

    G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.ones_like(D_gen_logits))

    return D_loss, G_loss

def _least_squares_loss(D_real, D_gen, D_real_logits=None, D_gen_logits=None):
    D_loss_real = torch.mean(torch.square(D_real - 1.0))
    D_loss_gen = torch.mean(torch.square(D_gen))
    D_loss = 0.5 * (D_loss_real + D_loss_gen)

    G_loss = 0.5 * torch.mean(torch.square(D_gen - 1.0))
    
    return D_loss, G_loss

def gan_loss(gan_loss_type, disc_out, mode='generator_loss'):

    if gan_loss_type == 'non_saturating':
        loss_fn = _non_saturating_loss
    elif gan_loss_type == 'least_squares':
        loss_fn = _least_squares_loss
    else:
        raise ValueError('Invalid GAN loss')

    D_loss, G_loss = loss_fn(D_real=disc_out.D_real, D_gen=disc_out.D_gen,
        D_real_logits=disc_out.D_real_logits, D_gen_logits=disc_out.D_gen_logits)
        
    loss = G_loss if mode == 'generator_loss' else D_loss
    
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.helpers import maths
from src.network import discriminator
from src.helpers.utils import get_scheduled_params

lower_bound_toward = maths.LowerBoundToward.apply

def weighted_rate_loss(config, total_nbpp, total_qbpp, step_counter, ignore_schedule=False, bypass_rate=False):
    """
    Heavily penalize the rate with weight lambda_A >> lambda_B if it exceeds 
    some target r_t, otherwise penalize with lambda_B
    """
    lambda_A = get_scheduled_params(config.lambda_A, config.lambda_schedule, step_counter, ignore_schedule)
    lambda_B = get_scheduled_params(config.lambda_B, config.lambda_schedule, step_counter, ignore_schedule)

    if bypass_rate is False:
        assert lambda_A > lambda_B, "Expected lambda_A > lambda_B, got (A) {} <= (B) {}".format(
            lambda_A, lambda_B)

    target_bpp = get_scheduled_params(config.target_rate, config.target_schedule, step_counter, ignore_schedule)

    total_qbpp = total_qbpp.item()
    if total_qbpp > target_bpp:
        rate_penalty = lambda_A
    else:
        rate_penalty = lambda_B

    if bypass_rate is True:
        # Loose rate
        weighted_rate = config.lambda_B * total_nbpp
        rate_penalty = config.lambda_B
    else:
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

def _linear_annealing(init, fin, step, annealing_steps):
    """
    Linear annealing of a parameter. Linearly increase parameter from
    value 'init' to 'fin' over number of iterations specified by
    'annealing_steps'
    """
    if annealing_steps == 0:
        return fin
    assert fin > init, 'Final value should be larger than initial'
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def TC_loss(latents, tc_discriminator, step_counter, model_training):

    half_batch_size = latents.size(0) // 2
    try:
        latents, latents_D = torch.split(latents, half_batch_size, dim=0)
    except ValueError:
        latents, latents_D, *_ = torch.split(latents, half_batch_size, dim=0)

    annealing_steps = 1e4
    anneal_reg = (_linear_annealing(0, 1, min(0, step_counter - 1e4), annealing_steps) 
            if model_training else 1)

    print('anneal_reg', anneal_reg)
    tc_disc_logits = tc_discriminator(latents)
    TC_term = tc_disc_logits.mean()
    # TC_term = (tc_disc_logits[:,0] - tc_disc_logits[:,1]).flatten().mean()
    TC_term = lower_bound_toward(TC_term, 0.)
    print('TC_term', TC_term.item())
    tc_loss = anneal_reg * TC_term
    print('tcL', tc_loss.item())

    latents_perm_D = maths._permute_dims_2D(latents_D.detach())
    tc_disc_logits_perm = tc_discriminator(latents_perm_D)

    #tc_loss_marginal = F.cross_entropy(input=tc_disc_logits, 
    #    target=torch.zeros(batch_size, dtype=torch.long, device=latents.device))
    #tc_loss_perm = F.cross_entropy(input=tc_disc_logits_perm,
    #    target=torch.ones(batch_size, dtype=torch.long, device=latents.device))
    tc_loss_marginal = F.binary_cross_entropy_with_logits(input=tc_disc_logits, 
        target=torch.zeros_like(tc_disc_logits))
    tc_loss_perm = F.binary_cross_entropy_with_logits(input=tc_disc_logits_perm,
        target=torch.ones_like(tc_disc_logits_perm))
    tc_disc_loss = 0.5 * (tc_loss_marginal + tc_loss_perm)
    print('tc_dL', tc_disc_loss.item())

    return tc_loss, tc_disc_loss

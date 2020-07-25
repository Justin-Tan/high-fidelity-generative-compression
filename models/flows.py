import numpy as np

from scipy import stats
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from utils import helpers, initialization, math, distributions, normalization



class DiscreteFlowModel(nn.Module):
    """ Allows density estimation of  data x = T(u) by computing 
        p(T^{-1}(x)) + log |det J_{T^{-1}}(x)|}. 
        Composes arbitrary flows. """

    def __init__(self, input_dim, hidden_dim, base_dist=None, n_flows=8,
                 flow=distributions.InvertibleAffineFlow, context_dim=0):
        super(DiscreteFlowModel, self).__init__()
        
        self.n_flows = n_flows
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.base_dist = base_dist
        
        BN_flow = normalization.BatchNormFlow
        # BN_flow = normalization.MovingBatchNorm1d
        parity = lambda n: True if n%2==0 else False
        
        # Aggregate parameters from each transformation in the flow
        for k in range(self.n_flows):
            flow_k = flow(input_dim=self.input_dim, parity=parity(k), hidden_dim=self.hidden_dim, context_dim=self.context_dim)
            # BN_k = BN_flow(input_dim=self.input_dim)
            BN_k = BN_flow(self.input_dim)
            self.add_module('flow_{}'.format(str(k)), flow_k)
            self.add_module('BN_{}'.format(str(k)), BN_k)

    def forward(self, x, invert=False, context=None):
        if invert:  # Density estimation training
            return self._backward(x, context)
        else:  # Sample from learned model
            return self._forward(x, context)

    def _forward(self, x_0, context=None):
        """ Sample from target density by passing x0 sampled from
            base distribution through forward flow, x = F(x_0). 
            Used for sampling from trained model. """

        batch_size = x_0.shape[0]
        x_flow = [x_0]  # Sequence of residual flows. \hat{x} = x_flow[-1]

        log_det_jacobian = torch.zeros(batch_size).to(x_0)

        for k in range(self.n_flows):
            
            x_k = x_flow[-1]
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.forward(x_k, context)
            
            # Don't apply batch norm after final forward flow T_{K-1}
            if k < self.n_flows: # - 1:
                BN_k = getattr(self, 'BN_{}'.format(str(k)))
                x_k, log_det_jacobian_BN_k = BN_k(x_k, logpx=torch.zeros(batch_size, 1).to(x_k))
                log_det_jacobian += torch.squeeze(log_det_jacobian_BN_k)
                
            x_flow.append(x_k)
            log_det_jacobian += log_det_jacobian_k

        # Final approximation of target sample
        x_K = x_flow[-1]

        flow_output = {'log_det_jacobian': log_det_jacobian, 'x_flow': x_flow}

        return flow_output 


    def _backward(self, x, context=None):
        """ Recover base x0 ~ N(\mu, \Sigma) by inverting 
            flow transformations: x0 = F^{-1}(x). Used for density 
            evaluation. """

        x_flow_inv = [x]
        batch_size = x.shape[0]
        log_det_jacobian_inv = torch.zeros(batch_size).to(x)

        # Sequence T^{-1}(x) = u ==> x -> x_{K-1} -> ... -> x_1 -> u
        for k in range(self.n_flows)[::-1]:  # reverse order
            
            x_k = x_flow_inv[-1]
            
            # Don't apply batch norm before transform T^{-1}_{K_1}
            if k < self.n_flows: # - 1:
                BN_k = getattr(self, 'BN_{}'.format(str(k)))
                x_k, log_det_jacobian_BN_k = BN_k(x_k, logpx=torch.zeros(batch_size, 1).to(x_k), reverse=True)
                log_det_jacobian_inv += torch.squeeze(log_det_jacobian_BN_k)

            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.invert(x_k, context)
        
            x_flow_inv.append(x_k)
            
            log_det_jacobian_inv += log_det_jacobian_k

        inv_flow_output = {'log_det_jacobian_inv': log_det_jacobian_inv, 'x_flow_inv': x_flow_inv}

        return inv_flow_output
    
    def log_density(self, x, context=None):
        
        batch_size = x.shape[0]
        inv_flow_output = self._backward(x, context)
        
        log_det_jacobian_inv = inv_flow_output['log_det_jacobian_inv']
        x_flow_inv = inv_flow_output['x_flow_inv']
        
        x_0 = x_flow_inv[-1]
        log_px_0 = self.base_dist.log_density(x_0, mu=torch.zeros_like(x_0), 
            logvar=torch.zeros_like(x_0)).view(batch_size, -1).sum(dim=1)
            
        log_px = log_px_0 + log_det_jacobian_inv
        return log_px
    
    def sample(self, x, context=None):
        
        batch_size = x.shape[0]
        x_0 = self.base_dist.sample(shape=x.shape, mu=torch.zeros_like(x), 
                logvar=torch.zeros_like(x)).cuda()
        
        flow_output = self._forward(x_0, context)
        log_det_jacobian = flow_output['log_det_jacobian']
        x_flow = flow_output['x_flow']
        x_K = x_flow[-1]
        
        log_px_0 = self.base_dist.log_density(x_0, mu=torch.zeros_like(x_0), 
            logvar=torch.zeros_like(x_0)).view(batch_size, -1).sum(dim=1)
        log_px_K = log_px_0 - log_det_jacobian
        
        return x_K, log_px_K

""" Using open-source material: 
    @rtiqchen 2018
    @YannDubs 2019
    @riannevdberg 2018 
    @karpathy 2019 """

import torch
import torch.nn as nn
import numpy as np

# self imports
from models import network
from utils import math, distributions, initialization, helpers

class VAE(nn.Module):
    
    def __init__(self, args, encoder_manual=None, decoder_manual=None):

        """
        Class which defines model and forward pass.
        Parameters
        ----------
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont'.
            (Not tested with only discrete).

        Specifying both `encoder_manual` and `decoder_manual` overrides the 
        preset encoder/decoder.
        """
        super(VAE, self).__init__()
        
        print(args)
        try:
            self.input_dim = args.input_dim
        except AttributeError:
            self.input_dim = args.im_shape

        try:
            self.hidden_dim = args.hidden_dim
        except AttributeError:
            self.hidden_dim = 256

        self.output_dim = np.prod(self.input_dim)
        self.latent_spec = args.latent_spec
        self.is_discrete = ('discrete' in self.latent_spec.keys())
        self.latent_dim = self.latent_spec['continuous']
        self.flow_output = {'log_det_jacobian': None, 'x_flow': None}

        if not hasattr(args, 'prior') or args.prior == 'normal':
            self.prior = distributions.Normal()
        elif args.prior == 'flow':
            self.prior = distributions.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=args.flow_steps)

        try:
            if args.x_dist == 'bernoulli':
                self.x_dist = distributions.Bernoulli()
            elif args.x_dist == 'normal':
                self.x_dist = distributions.Normal()
        except AttributeError:
            self.x_dist = distributions.Bernoulli()

        if self.is_discrete:
            assert sum(self.latent_spec['discrete']) > 0, 'Must have nonzero number of discrete latent dimensions.'
            print('Using discrete latent factors with specification:', self.latent_spec['discrete'])
            self.latent_dim_discrete = sum([dim for dim in self.latent_spec['discrete']])
            self.n_discrete = len(self.latent_spec['discrete'])
            self.latent_dim += self.latent_dim_discrete  # OK, not 100% consistent

        if args.mlp is True:
            assert args.dataset != 'custom', 'Custom option incompatiable with mlp option!'
            encoder = network.MLPEncoder
            decoder = network.MLPDecoder
        else:
            encoder = network.EncoderVAE_conv
            decoder = network.DecoderVAE_conv

        if args.dataset in ['custom', 'jets']:
            encoder = network.ToyEncoder
            decoder = network.ToyDecoder

        if encoder_manual is not None and decoder_manual is not None:
            # Manual override
            encoder = encoder_manual
            decoder = decoder_manual

        self.encoder = encoder(input_dim=self.input_dim, latent_spec=self.latent_spec, hidden_dim=self.hidden_dim)
        self.decoder = decoder(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,
            output_dim=self.output_dim)
        self.reset_parameters()

        print('Using prior:', self.prior)
        print('Using likelihood p(x|z):', self.x_dist)
        print('Using posterior p(z|x): Diagonal-covariance Gaussian')
        
    def reset_parameters(self):
        self.apply(initialization.weights_init)

    def reparameterize(self, latent_stats):
        """
        Combines continuous and discrete latent samples.
        Parameters
        ----------
        latent_stats : dict, torch.Tensor values
            Dict containing at least one of the keys 'continuous' or 'discrete'
            containing the parameters of the latent distributions as torch.Tensor 
            instances.
        """
        mu, logvar = latent_stats['continuous']
        latent_sample = [self.reparameterize_continuous(mu, logvar)]

        if self.is_discrete:
            alphas = latent_stats['discrete']
            discrete_sample = [self.reparameterize_discrete(alpha) for alpha in alphas]
            latent_sample += discrete_sample

        latent_sample = torch.cat(latent_sample, dim=1)

        return latent_sample            
        
        
    def reparameterize_continuous(self, mu, logvar):
        """
        Sample from N(mu(x), Sigma(x)) as 
        z | x ~ mu + Cholesky(Sigma(x)) * eps
        eps ~ N(0,I_n)
        
        The variance is restricted to be diagonal,
        so Cholesky(...) -> sqrt(...)
        Parameters
        ----------
        mu     : torch.Tensor
            Location parameter of Gaussian. (B, latent_dim)

        logvar : torch.Tensor
            Log of variance parameter of Gaussian. (B, latent_dim)

        """
        if self.training:
            sigma = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(sigma)
            return mu + sigma * epsilon
        else:
            # Reconstruction, return mean
            return mu
        
    def reparameterize_discrete(self, alpha, temperature=0.67):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (B, latent_dim)
        """
        EPS = 1e-12
        
        
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).to(alpha.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / temperature
            return F.softmax(logit, dim=1)
        else:
            # Reconstruction, return most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            
            one_hot_samples = one_hot_samples.to(alpha.device)
            return one_hot_samples
    
    def forward(self, x, **kwargs):
        latent_stats = self.encoder(x)
        mu_z, logvar_z = latent_stats['continuous']
        latent_sample = self.reparameterize(latent_stats)
        x_stats = self.decoder(latent_sample)
        
        return x_stats, latent_sample, latent_stats, self.flow_output


class VAE_ODE(VAE):
    """ Subclass of VAE - replaces decoder with continuous normalizing flow, with
        dynamics determined by a neural network, which is a function of z ~ q(z|x). 

        Sample x_0 from base distribution emitted by generative network
        x_0 ~ p(x_0 | z)
        x_0 -> CNF -> \hat{x}

        Identical encoder logic to standard VAE. Allows density estimation of 
        data x by computing the change in log-density via numerical integration
        by black-box ODE solver. """

    def __init__(self, args):
        super(VAE_ODE, self).__init__(args)
        assert args.flow in ['cnf', 'cnf_freeze_vae'] , 'Must toggle CNF option in arguments!'

        dims = self.input_dim
        self.flow = args.flow
        self.cnf = build_model_tabular(args, dims)
        set_cnf_options(args, self.cnf)

    def _get_transforms(self, model):

        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def forward(self, x, sample=False):

        flow_output = {'log_det_jacobian': None, 'x_flow': None}
        sample_fn, density_fn = self._get_transforms(self.cnf)

        def _vae_forward(x):
            latent_stats = self.encoder(x)
            latent_sample = self.reparameterize(latent_stats)
            # Parameters of base distribution - diagonal covariance Gaussian
            x_stats = self.decoder(latent_sample)

            return latent_stats, latent_sample, x_stats

        if self.flow == 'cnf_freeze_vae':
            with torch.no_grad():
                latent_stats, latent_sample, x_stats = _vae_forward(x)
        else:
            latent_stats, latent_sample, x_stats = _vae_forward(x)

        if sample is True:
            # Return reconstructed sample from target density, reverse pass
            # x_0 -> CNF^{-1} -> x_K
            with torch.no_grad():
                x0_sample = self.reparameterize_continuous(mu=x_stats['mu'], logvar=x_stats['logvar'])
                x_hat = sample_fn(x0_sample)  # reverse pass
                flow_output['x_flow'] = x_hat

        else:
            # Invert CNF to fit flow-based model to target density, by 
            # transforming x to sample x_0 from base distribution
            # x_K -> CNF -> x_0
            zero = torch.zeros(x.shape[0], 1).to(x)
            x_0, delta_logp = self.cnf(x, zero)
            flow_output['x_flow'] = x_0
            flow_output['log_det_jacobian'] = delta_logp  #.view(-1)

        return x_stats, latent_sample, latent_stats, flow_output
    
def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.input_dim,)

class AmortizedLowRankODEnet(nn.Module):

    def __init__(self, hidden_dims, input_dim, rank=1, layer_type="concat", nonlinearity="softplus"):
        super(AmortizedLowRankODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        self.input_dim = input_dim

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_dim

        # Input dimensions: (x_dim, hidden_dim_1, ..., hidden_dim_N)
        # Output dimensions: (hidden_dim_1, ..., hidden_dim_N, x_dim)
        self.output_dims = hidden_dims
        self.input_dims = (input_dim,) + hidden_dims[:-1]

        for dim_out in hidden_dims:
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.rank = rank

    def _unpack_params(self, params):
        return [params]

    def _rank_k_bmm(self, x, u, v):

        # bmm: z = torch.bmm(x,y) | x: (b, m, p), y: (b, p, n), z: (b, m, n))
        # x: (b, 1, D_in), u: (b, D_in, k), v: (b, k, D_out)
        xu = torch.bmm(x[:, None], u.view(x.shape[0], x.shape[-1], self.rank))  # (b, 1, k)
        xuv = torch.bmm(xu, v.view(x.shape[0], self.rank, -1))  # (b, 1, D_out)
        return xuv[:, 0]

    def forward(self, t, y, am_params):
        dx = y
        for l, (layer, in_dim, out_dim) in enumerate(zip(self.layers, self.input_dims, self.output_dims)):
            # am_params shape: (D_in + D_out) * rank + D_out
            this_u, am_params = am_params[:, :in_dim * self.rank], am_params[:, in_dim * self.rank:]
            this_v, am_params = am_params[:, :out_dim * self.rank], am_params[:, out_dim * self.rank:]
            this_bias, am_params = am_params[:, :out_dim], am_params[:, out_dim:]

            # Previous output becomes current input
            xw = layer(t, dx)
            xw_am = self._rank_k_bmm(dx, this_u, this_v)
            dx = xw + xw_am + this_bias
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


def construct_amortized_odefunc(args, input_dim):

    hidden_dims = get_hidden_dims(args)

    diffeq = AmortizedLowRankODEnet(
            hidden_dims=hidden_dims,
            input_dim=input_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            rank=args.rank,
        )

    odefunc = layers.ODEfunc(
        diffeq=diffeq,
        divergence_fn=args.divergence_fn,
        residual=args.residual,
        rademacher=args.rademacher,
    )

    return odefunc


class VAE_ODE_amortized(VAE):
    
    """ Subclass of VAE - replaces decoder with continuous normalizing flow, with
        dynamics determined by a neural network, which is a function of z ~ q(z|x). 
        Performs amortized variational inference; parameters of dynamics network are 
        a function of z.

        Sample x_0 from base distribution emitted by generative network
        x_0 ~ p(x_0 | z)
        x_0 -> CNF (with input-dependent dynamics) -> \hat{x}

        Identical encoder logic to standard VAE. Allows density estimation of 
        data x by computing the change in log-density via numerical integration
        by black-box ODE solver. 

        Variational inference is amortized. Instead of learning the parameters of the posterior
        distribution for each data point, the input dependence of the posterior distribution
        parameters is modelled through an encoder/decoder network. The flow parameters are 
        treated as functions of the original datapoint. In the case of CNFs, this corresponds
        to treating the parameters of the layers defining the flow dynamics,

        dx/dt = f(x; \theta; t)

        As functions of x themselves. Practically, the encoder/decoder network outputs a low-
        rank matrix input-dependent update to a global weight matrix, and outputs an input-
        dependence bias vector update to a global bias term. See equation 10 in [1].

        [1]: FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models
             Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, David Duvenaud
    """

    def __init__(self, args):
        super(VAE_ODE_amortized, self).__init__(args)
        assert args.flow == 'cnf_amort', 'Must toggle amortized CNF option in arguments!'

        dims = self.input_dim

        # CNF model
        self.odefuncs = nn.ModuleList([
            construct_amortized_odefunc(args, args.input_dim) for _ in range(args.num_blocks)
        ])
        self.q_am = self._amortized_layers(args)
        assert len(self.q_am) == args.num_blocks or len(self.q_am) == 0

        # If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
        self.register_buffer('integration_times', torch.tensor([0.0, args.time_length]))

        self.atol = args.atol
        self.rtol = args.rtol
        self.solver = args.solver


    def _amortized_layers(self, args):
        # e.g. hidden sequence size (D, D), input dim N, update rank k
        # Note first input dim = final output dim
        out_dims = get_hidden_dims(args)  # (D, D, N)
        in_dims = (out_dims[-1],) + out_dims[:-1]  # (N, D, D)
        params_size = (sum(in_dims) + sum(out_dims)) * args.rank + sum(out_dims)
        return nn.ModuleList([nn.Linear(self.hidden_dim, params_size) for _ in range(args.num_blocks)])

    def _get_transforms(self, model):

        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def forward(self, x, sample=False):

        flow_output = {'log_det_jacobian': None, 'x_flow': None}
        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)

        # Parameters of base distribution - diagonal covariance Gaussian
        x_stats = self.decoder(latent_sample)
        x_0 = self.reparameterize_continuous(mu=x_stats['mu'], logvar=x_stats['logvar'])
        # Amortized input-dependent flow parameters
        h = x_stats['hidden'].view(-1, self.hidden_dim)
        am_params = [q_am(h) for q_am in self.q_am]

        delta_logp = torch.zeros(x.shape[0], 1).to(x)
        y = x_0

        def cnf_forward_pass(y, delta_logp, am_params):
            for odefunc, am_param in zip(self.odefuncs, am_params):
                am_param_unpacked = odefunc.diffeq._unpack_params(am_param)
                odefunc.before_odeint()
                states = odeint(
                    odefunc,
                    (y, delta_logp) + tuple([torch.ones_like(am_param)]), # tuple(am_param_unpacked),
                    self.integration_times.to(y),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.solver,
                )
                y, delta_logp = states[0][-1], states[1][-1]

            return y, delta_logp

        if sample is True:
            with torch.no_grad():
                y, delta_logp = cnf_forward_pass(y, delta_logp, am_params)
        else:
            y, delta_logp = cnf_forward_pass(y, delta_logp, am_params)

        x_hat = y

        flow_output['x_flow'] = x_hat
        flow_output['log_det_jacobian'] = -delta_logp

        return x_stats, latent_sample, latent_stats, flow_output


class discrete_flow_VAE(VAE):
    """ Subclass of VAE - implements a sequence of invertible normalizing flows on
        top of the base Gaussian decoder.
        Identical encoder logic to standard VAE. Allows density estimation of 
        data x = T(u) by computing p(T^{-1}(x)) + log |det J_{T^{-1}}(x)|}. """

    def __init__(self, args):
        super(discrete_flow_VAE, self).__init__(args)
        self.n_flows = args.flow_steps
        self.input_dim = args.input_dim
        self.hidden_dim = args.discrete_flow_hidden_dim

        # TODO: Expand possible invertible flows 
        self.flow = distributions.DiscreteFlowModel(input_dim=self.input_dim,
            hidden_dim=self.hidden_dim, n_flows=self.n_flows,
            flow=distributions.InvertibleAffineFlow)
            

    def forward(self, x, sample=False):
        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)

        # Parameters of base distribution - diagonal covariance Gaussian
        x_stats = self.decoder(latent_sample)
        # Sample from base distribution
        x0 = self.reparameterize_continuous(mu=x_stats['mu'], logvar=x_stats['logvar'])

        if sample is True:
            # Return reconstructed sample from target density
            with torch.no_grad():
                # flow_output = self.forward_flow(x_stats)
                flow_output = self.flow.forward(x0)
                x_hat = flow_output['x_flow'][-1]
                return x_stats, latent_sample, latent_stats, flow_output
        else:
            # Invert flow to fit flow-based model to target density
            # Note reversed order of flow, LDJ in backward output
            # inv_flow_output = self.backward_flow(x)
            inv_flow_output = self.flow.backward(x)
            x0 = inv_flow_output['x_flow_inv'][-1]
            return x_stats, latent_sample, latent_stats, inv_flow_output


class PlanarVAE(VAE):
    """ Subclass of VAE - implements planar flows in the encoder.
        Identical decoder logic to standard VAE, changes encoder
        to implement z_0 -> z_1 -> ... -> z_K flow. """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)
        
        flow = distributions.PlanarFlow
        self.n_flows = args.flow_steps

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.hidden_dim, self.n_flows * self.latent_dim)
        self.amor_w = nn.Linear(self.hidden_dim, self.n_flows * self.latent_dim)
        self.amor_b = nn.Linear(self.hidden_dim, self.n_flows)

        # Normalizing flow layers
        for k in range(self.n_flows):
            flow_k = flow(dim=self.latent_dim)
            self.add_module('flow_{}'.format(str(k)), flow_k)

        self.planar_flow = flow(dim=self.output_dim)


    def forward(self, x):
        """
        Normalizing flow in decoder implements more flexible likelihood term to handle 
        non-Gaussian densities.
        """
        batch_size = x.shape[0]

        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)
        z_0 = latent_sample

        z_flow = [z_0]  # Sequence of residual flows. \hat{x} = x_flow[-1]
        h = latent_stats['hidden']  # activation before projection into mu, logvar

        # return amortized u/w/b for all flows
        u = self.amor_u(h).view(batch_size, self.n_flows, self.latent_dim, 1)
        w = self.amor_w(h).view(batch_size, self.n_flows, 1, self.latent_dim)
        b = self.amor_b(h).view(batch_size, self.n_flows, 1, 1)

        log_det_jacobian = torch.zeros([batch_size, self.n_flows], requires_grad=True).type_as(x.data)

        for k in range(self.n_flows):
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            z_k, log_det_jacobian_k = flow_k(z_flow[k], u[:,k,:,:], w[:,k,:,:], b[:,k,:,:])
            # x_k, log_det_jacobian_k = self.planar_flow(x_flow[k], u[:,k,:,:], w[:,k,:,:], b[:,k,:,:])
            z_flow.append(z_k)
            log_det_jacobian[:,k] = log_det_jacobian_k
            # log_det_jacobian += log_det_jacobian_k

        # Final approximation of target density
        z_K = z_flow[-1]

        # Parameters of output distribution
        x_stats = self.decoder(z_K)

        self.flow_output = {'log_det_jacobian': log_det_jacobian, 'z_flow': z_flow}

        return x_stats, latent_sample, latent_stats, self.flow_output



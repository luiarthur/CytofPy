import copy
import math

import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta, Bernoulli
from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence as kld
from cytopy.model.vardist import VDGamma, VDNormal, VDBeta, VDDirichlet, VDLogNormal, VI

from torch.nn import Parameter, ParameterList

def compute_Z(logit, tau):
    """
    This enables the backward gradient computations via Z.
    Notice at the end, we basically return Z (binary tensor).
    But we make use of the smoothed Z, which is differentiable.
    We detach (Z - smoothed_Z) so that the gradients are not 
    computed, and then add back smoothed_Z for the return.
    The only gradient will then be that of smoothed Z.
    """
    smoothed_Z = (logit / tau).sigmoid()
    Z = (smoothed_Z > 0.5).float()
    return (Z - smoothed_Z).detach() + smoothed_Z


def default_priors(y, K:int=30, L=None):
    I = len(y)

    J = y[0].size(1)
    for i in range(I):
        assert(y[i].size(1) == J)

    N = [y[i].size(0) for i in range(I)]

    K = K

    if L is None:
        L = [5, 5]

    return {'I': I, 'J': J, 'N': N, 'L': L, 'K': K,
            #
            'delta0': Gamma(1, 1),
            'delta1': Gamma(1, 1),
            #
            'sig0': LogNormal(0, 1),
            'sig1': LogNormal(0, 1),
            #
            'eta0': Dirichlet(torch.ones(L[0]) / L[0]),
            'eta1': Dirichlet(torch.ones(L[1]) / L[1]),
            #
            'alpha': Gamma(.1, .1),
            'H': Normal(0, 1),
            #
            'b0': Gamma(1, 1),
            'b1': Gamma(1, 1),
            #
            'W': Dirichlet(torch.ones(K) / K)}

def initialize_y_vd(y, m=None, mean_init=-6.0, sd_init=1.0):
    """
    make sure to zero the gradients for observed y
    """
    I = len(y)

    if m is None:
        m = [torch.isnan(yi) for yi in y]

    # Copy all. y_vp_m for observed will then be the observed values.
    y_vp_m_init = copy.deepcopy(y)

    # Set sd for observed values to 0, so sampling yields the observed values.
    y_vp_log_s_init = [torch.ones(yi.shape) * float('-inf') for yi in y]

    for i in range(I):
        # Set means for missing y to mean_init
        y_vp_m_init[i][m[i]] = mean_init

        # Set log_sd for missing y to log(sd_init)
        y_vp_log_s_init[i][m[i]] = math.log(sd_init)

    y_vd = [VDNormal(y[i].shape, y_vp_m_init[i], y_vp_log_s_init[i]) for i in range(I)]

    return y_vd


class Model(VI):
    def __init__(self, y, priors, m=None, y_mean_init=-6.0, y_sd_init=0.5,
                 tau=0.1, verbose=1):
        self.verbose = verbose

        # Dimensions of data
        self.I = priors['I']
        self.J = priors['J']
        self.N = priors['N']
        self.Nsum = sum(self.N)

        # Tuning Parameters
        self.tau = tau

        # Dimensions of parameters
        self.L = priors['L']
        self.K = priors['K']

        # Store priors
        self.priors = priors

        # Assign variational distributions
        self.delta0 = VDGamma(self.L[0])
        self.delta1 = VDGamma(self.L[1])
        self.sig0 = VDLogNormal(self.L[0])
        self.sig1 = VDLogNormal(self.L[1])
        # self.sig = VDLogNormal(self.I)
        self.eta0 = VDDirichlet((self.I, self.J, self.L[0]))
        self.eta1 = VDDirichlet((self.I, self.J, self.L[1]))
        self.W = VDDirichlet((self.I, self.K))
        self.alpha = VDGamma(1)
        self.b0 = VDGamma((self.I, self.J))
        self.b1 = VDGamma((self.I, self.J))
        self.v = VDBeta(self.K)
        self.H = VDNormal((self.J, self.K))

        # This must be done after assigning variational distributions
        super(Model, self).__init__()

        # Register m
        self.m = [torch.isnan(yi) for yi in y]

        # Record the variational distribution for y
        self.y = initialize_y_vd(y=y, m=m, mean_init=y_mean_init, sd_init=y_sd_init)
        self.vd['y'] = self.y
        # Register Variational Parameters for y
        self.y_vp = ParameterList(self.y[i].vp for i in range(self.I))
        

    def loglike(self, data, params):
        idx = data['idx']
        y = params['y']
        m = [self.m[i][idx[i], :] for i in range(self.I)]

        ll = 0.0
        for i in range(self.I):
            # Y: Ni x J
            # muz: Lz
            # etaz_i: 1 x J x Lz

            # Ni x J x Lz
            mu0 = -params['delta0'].cumsum(0)
            d0 = Normal(mu0[None, None, :],
                        # params['sig'][i]).log_prob(y[i][:, :, None])
                        params['sig0'][None, None, :]).log_prob(y[i][:, :, None])
            d0 += params['eta0'][i:i+1, :, :].log()

            mu1 = params['delta1'].cumsum(0)
            d1 = Normal(mu1[None, None, :],
                        # params['sig'][i]).log_prob(y[i][:, :, None])
                        params['sig1'][None, None, :]).log_prob(y[i][:, :, None])
            d1 += params['eta1'][i:i+1, :, :].log()
            
            # Ni x J
            logmix_L0 = torch.logsumexp(d0, 2)
            logmix_L1 = torch.logsumexp(d1, 2)

            # Z: J x K
            # H: J x K
            # v: K
            # c: Ni x J x K
            # d: Ni x K
            # Ni x J x K

            b_vec = params['v'].cumprod(0)
            Z = compute_Z(b_vec[None, :] - Normal(0, 1).cdf(params['H']), self.tau)
            c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
            d = c.sum(1)

            fac = self.N[i] / self.Nsum 

            f = d + params['W'][i:i+1, :].log()
            lli = torch.logsumexp(f, 1).mean(0)

            logit_pi = -params['b0'][i:i+1, :] - params['b1'][i:i+1, :] * y[i]
            lli += Bernoulli(logits=logit_pi).log_prob(m[i].float()).sum(1).mean(0)

            assert(lli.dim() == 0)

            ll += lli * fac

        if self.verbose >= 2:
            print('log_like: {}'.format(ll))

        return ll

    def kl_qp(self, params, idx):
        res = 0.0

        for key in params:
            if key == 'v':
                res += kld(self.vd['v'].dist(), Beta(params['alpha'], 1)).sum()
            elif key == 'y':
                for i in range(self.I):
                    mi = self.m[i][idx[i], :]
                    y_vp_m = self.y[i].vp[0][idx[i], :][mi]
                    y_vp_s = self.y[i].vp[1][idx[i], :][mi].exp()
                    yi = params['y'][i]
                    res -= Normal(y_vp_m, y_vp_s).log_prob(yi[mi]).sum()
            else:
                # NOTE: self.vd refers to the variational distributions object
                res += kld(self.vd[key].dist(), self.priors[key]).sum()

        if self.verbose >= 2:
            print('kl_qp: {}'.format(res / self.Nsum))

        return res / self.Nsum

    def sample_params(self, idx):
        params = {}
        for key in self.vd:
            if key == 'y':
                params['y'] = []
                for i in range(self.I):
                    yi_vp = self.y_vp[i][:, idx[i], :]
                    yi = Normal(yi_vp[0], yi_vp[1].exp()).rsample()
                    params['y'].append(yi)
            else:
                params[key] = self.vd[key].rsample()

        return params

    def forward(self, data):
        idx = data['idx']
        params = self.sample_params(idx)
        elbo = self.loglike(data, params) - self.kl_qp(params, idx)
        return elbo

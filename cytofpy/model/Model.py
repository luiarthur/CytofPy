import copy
import math

import torch

from torch.distributions.log_normal import LogNormal
from torch.distributions import Normal, Gamma, Dirichlet, Beta, Bernoulli
from torch.nn import ParameterList

from cytofpy.model.model_param import ModelParam, VI

import numpy as np

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

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
    Z = (smoothed_Z > 0.5).double()
    return (Z - smoothed_Z).detach() + smoothed_Z

@torch.jit.script
def prob_miss_logit(y, b0, b1, b2):
    return b0 + b1 * y + b2 * y**2

@torch.jit.script
def prob_miss(y, b0, b1, b2):
    return prob_miss_logit(y, b0, b1, b2).sigmoid()

def solve_beta(y, p):
    k = len(p)
    p = np.array(p)
    Y = np.concatenate([[np.ones(k)], [y], [y**2]]).T
    beta = np.linalg.solve(Y, np.log(p) - np.log1p(-p))
    return torch.tensor(beta).double()

# TODO: Put this test in tests/
# solve_beta(np.array([-5.0, -3.0, -1.0]), np.array([.05, .8 , .05]))
# exact: -8.35785565, -6.49610001, -1.08268334

def gen_beta_est(y_ij, y_quantiles, p_bounds):
    yij_neg = y_ij[y_ij < 0]
    y_bounds = np.percentile(yij_neg.numpy(), y_quantiles)
    return solve_beta(y_bounds, p_bounds)

# TODO: Put this test in tests/
# gen_beta_est(torch.randn(10000), [3, 30, 50], [.05, .8, .05])
# approx: -18.763069081151038 -30.50605414798636 -10.654946017898444

def default_priors(y, K:int=30, L=None,
                   y_quantiles=[0, 35, 70], p_bounds=[.05, .8, .05], y_bounds=None):
    I = len(y)

    J = y[0].size(1)
    for i in range(I):
        assert(y[i].size(1) == J)

    N = [y[i].size(0) for i in range(I)]

    K = K

    if L is None:
        L = [5, 5]
    
    b0 = torch.zeros((I, J))
    b1 = torch.zeros((I, J))
    b2 = torch.zeros((I, J))

    for i in range(I):
        for j in range(J):
            if y_bounds is None:
                beta = gen_beta_est(y[i][:, j], y_quantiles, p_bounds)
            else:
                beta = solve_beta(np.array(y_bounds), p_bounds)

            b0[i, j] = beta[0]
            b1[i, j] = beta[1]
            b2[i, j] = beta[2]


    return {'I': I, 'J': J, 'N': N, 'L': L, 'K': K,
            #
            'delta0': Gamma(1, 1),
            'delta1': Gamma(1, 1),
            #
            # 'sig0': LogNormal(0, 1),
            # 'sig1': LogNormal(0, 1),
            'sig': LogNormal(0, 1),
            #
            'eta0': Dirichlet(torch.ones(L[0]) / L[0]),
            'eta1': Dirichlet(torch.ones(L[1]) / L[1]),
            #
            'alpha': Gamma(.1, .1),
            'H': Normal(0, 1),
            #
            'b0': b0,
            'b1': b1,
            'b2': b2,
            #
            'W': Dirichlet(torch.ones(K) / K)}

def init_y_mp(y, m=None, mean_init=-4.0, sd_init=0.5):
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

    y_mp = [ModelParam(y[i].shape, 'real', m=y_vp_m_init[i], log_s=y_vp_log_s_init[i])
            for i in range(I)]

    return y_mp


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
        # coefficients defining the missing mechanism
        self.b0 = priors['b0']
        self.b1 = priors['b1']
        self.b2 = priors['b2']

        # Dimensions of parameters
        self.L = priors['L']
        self.K = priors['K']

        # Store priors
        self.priors = priors

        # Register m
        if m is None:
            self.m = [torch.isnan(yi) for yi in y]
        else:
            self.m = m

        self.msum = [mi.sum() for mi in self.m]

        ### Assign Model Parameters###
        self.mp = {}
        self.mp['delta0'] = ModelParam(self.L[0], 'positive')
        self.mp['delta1'] = ModelParam(self.L[1], 'positive')

        # self.mp['sig0'] = ModelParam(self.L[0], 'positive')
        # self.mp['sig1'] = ModelParam(self.L[1], 'positive')
        self.mp['sig'] = ModelParam(self.I, 'positive')

        self.mp['eta0'] = ModelParam((self.I, self.J, self.L[0] - 1), 'simplex')
        self.mp['eta1'] = ModelParam((self.I, self.J, self.L[1] - 1), 'simplex')
        self.mp['W'] = ModelParam((self.I, self.K - 1), 'simplex')
        self.mp['alpha'] = ModelParam(1, 'positive')
        self.mp['v'] = ModelParam(self.K, 'unit_interval')
        self.mp['H'] = ModelParam((self.J, self.K), 'real')
        self.mp['y'] = init_y_mp(y=y, m=self.m, mean_init=y_mean_init, sd_init=y_sd_init)
        ### END OF Assign Model Parameters###

        # This must be done after assigning model parameters
        super(Model, self).__init__()
        self.y_vp = ParameterList(mp_yi.vp for mp_yi in self.mp['y'])
        
    def vp_list(self):
        vp = []
        for key in self.mp:
            if key == 'y':
                for yi in self.mp['y']:
                    vp.append(yi.vp)
            else:
                vp.append(self.mp[key].vp)

        return vp

    def loglike(self, params, idx):
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
                        params['sig'][i]).log_prob(y[i][:, :, None])
                        # params['sig0'][None, None, :]).log_prob(y[i][:, :, None])
            d0 += params['eta0'][i:i+1, :, :].log()

            mu1 = params['delta1'].cumsum(0)
            d1 = Normal(mu1[None, None, :],
                        params['sig'][i]).log_prob(y[i][:, :, None])
                        # params['sig1'][None, None, :]).log_prob(y[i][:, :, None])
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

            # TODO: USE THIS FOR STICK-BREAKING IBP
            b_vec = params['v'].cumprod(0)
            # b_vec = params['v']
            Z = compute_Z(b_vec[None, :] - Normal(0, 1).cdf(params['H']), self.tau)
            c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
            d = c.sum(1)

            f = d + params['W'][i:i+1, :].log()

            fac = self.N[i] / self.Nsum 
            lli = torch.logsumexp(f, 1).mean(0) * fac

            assert(lli.dim() == 0)

            ll += lli

        if self.verbose >= 2:
            print('log_like: {}'.format(ll))

        return ll

    def log_q(self, reals, idx):
        out = 0.0
        for key in reals:
            if key == 'y':
                for i in range(self.I):
                    mi = self.m[i][idx[i], :]
                    if mi.sum() > 0:
                        y_vp_m = self.mp['y'][i].vp[0, idx[i], :][mi]
                        y_vp_s = self.mp['y'][i].vp[1, idx[i], :][mi].exp()
                        yi = reals['y'][i]
                        lq_yi = Normal(y_vp_m, y_vp_s).log_prob(yi[mi]).mean()

                        fac = self.msum[i] 
                        out += lq_yi * fac
            else:
                out += self.mp[key].log_q(reals[key])

        if self.verbose >= 2:
            print('log_q: {}'.format(out / self.Nsum))

        return out / self.Nsum

    def log_prior(self, reals, params, idx):
        out = 0.0
        for key in reals:
            if key == 'y':
                for i in range(self.I):
                    mi = self.m[i][idx[i], :]
                    if mi.sum() > 0:
                        pm_i = prob_miss(reals['y'][i],
                                         self.b0[i:i+1, :],
                                         self.b1[i:i+1, :],
                                         self.b2[i:i+1, :])
                        lp_yi = pm_i[mi].log().mean()

                        fac = self.msum[i] 
                        out += lp_yi * fac
            elif key == 'v':
                # TODO: USE STICK-BREAKING IBP
                tmp = Beta(params['alpha'], 1).log_prob(params['v'])
                # tmp = Beta(params['alpha'] / self.K, 1).log_prob(params['v'])
                tmp += self.mp['v'].logabsdetJ(reals['v'], params['v'])
                out += tmp.sum()
            else:
                tmp = self.priors[key].log_prob(params[key])
                tmp += self.mp[key].logabsdetJ(reals[key], params[key])
                out += tmp.sum()

        if self.verbose >= 2:
            print('log_prior: {}'.format(out / self.Nsum))

        return out / self.Nsum

    def sample_reals(self, idx):
        reals = {}
        for key in self.mp:
            if key == 'y':
                reals['y'] = []
                for i in range(self.I):
                    mi = self.m[i][idx[i], :].double()
                    yi_vp = self.mp['y'][i].vp[:, idx[i], :]
                    yi = Normal(yi_vp[0], yi_vp[1].exp()).rsample()
                    # NOTE: A trick to prevent computation of gradients for
                    #       observed values
                    yi = mi * yi + (1 - mi) * yi.detach()
                    reals['y'].append(yi)
            else:
                reals[key] = self.mp[key].real_sample()
                if self.mp[key].support in ['simplex', 'unit_interval']:
                    # NOTE: This prevents nan's in elbo and gradients.
                    #       This should not influence inference.
                    reals[key] = reals[key].clamp(min=-20, max=20)
                    if self.verbose >= 2:
                        print('WARNING: Clamping real {} to have magnitude of 20!'.format(key))

        return reals

    def transform(self, reals):
        params = {}
        for key in self.mp:
            if key == 'y':
                params['y'] = reals['y']
            else:
                params[key] = self.mp[key].transform(reals[key])
        return params

    def sample_params(self, idx):
        """
        used for post processing
        """
        reals = self.sample_reals(idx)
        return self.transform(reals)

    def forward(self, idx):
        reals = self.sample_reals(idx)
        params = self.transform(reals)
        elbo = self.loglike(params, idx) + self.log_prior(reals, params, idx)
        elbo -= self.log_q(reals, idx)
        return elbo

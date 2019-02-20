import abc
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.distributions import constraints

from torch.distributions.transforms import StickBreakingTransform

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

# Stick break transform function
sbt = StickBreakingTransform(0)

def has_constraint(prior, constraint):
    return isinstance(prior.support, type(constraint))

class ModelParam(abc.ABC):
    def __init__(self, size, prior, m=None, log_s=None):
        if m is None:
            m = torch.randn(size)

        if log_s is None:
            log_s = torch.randn(size)

        self.vp = torch.stack([m, log_s], requires_grad=True)
        self.size = size
        self.prior = prior

    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())

    def real_sample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def transform(self, real):
        if has_constraint(self.prior, constraints.unit_interval):
            return real.sigmoid()

        elif has_constraint(self.prior, constraints.simplex):
            return sbt(real)

        elif has_constraint(self.prior, constraints.positive):
            return real.exp()

        elif has_constraint(self.prior, constraints.real):
            return real

        else:
            NotImplemented

    def log_prior_plus_logabsdetJ(self, real, param):
        """
        This is most likely to be overriden in cases where
        the parameter is dependent on other parameters.
        """
        if has_constraint(self.prior, constraints.unit_interval):
            logabsdetJ = torch.log(param) + torch.log1p(-param)

        elif has_constraint(self.prior, constraints.simplex):
            logabsdetJ = sbt.log_abs_det_jacobian(real, param)

        elif has_constraint(self.prior, constraints.positive):
            logabsdetJ = real

        elif has_constraint(self.prior, constraints.real):
            logabsdetJ = torch.zeros(0)

        else:
            NotImplemented

        return self.prior.log_prob(param).sum() + logabsdetJ.sum()

    def log_q(self, real):
        return self.dist().log_prob(real).sum()


# TODO: ADD TESTS
# from torch.distributions import Gamma, Normal, Beta, Uniform, Dirichlet
# x = Param(1, Gamma(2,3))
# x = Param((3,5), Normal(-2,3))
# x = Param(3, Beta(2,3))
# conc = torch.tensor([2,2,1.])
# x = Param((3,5,2), Dirichlet(conc))
# 
# r = x.real_sample()
# p = x.transform(r)
# x.log_prior_plus_logabsdetJ(r, p)
# 
# x.log_q(x.real_sample())

class VI(abc.ABC):
    def __init__(self):
        self.mp = {}

    def vp_list(self):
        return [mp.vp for mp in mp.values()]

    @abc.abstractmethod
    def loglike(self, data, params):
        pass

    def log_prior(self, reals, params):
        out = 0.0
        for key in reals:
            out += self.mp[key].log_prior(reals[key], params[key])
        return out

    def log_q(self, reals):
        out = 0.0
        for key in reals:
            out += self.mp[key].log_q(reals[key])
        return out

    def sample_reals(self):
        reals = {}
        for key in self.mp:
            reals[key] = self.mp[key].real_sample()
        return reals
   
    def transform(self, reals):
        params = {}
        for key in self.mp:
            params[key] = self.mp[key].transform(reals[key])
        return params
 
    def compute_elbo(self, data):
        reals = self.sample_reals()
        params = self.transform(reals)
        elbo = self.loglike(data, params) + log_prior(params, reals)
        elbo -= log_q(reals)
        return elbo

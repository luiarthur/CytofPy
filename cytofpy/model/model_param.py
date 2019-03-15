import abc

import torch
from torch.distributions import Normal
from torch.distributions.transforms import StickBreakingTransform
from torch.nn import Parameter

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

# Stick break transform function
sbt = StickBreakingTransform(0)

# For checking constraints of model parameters
def is_unit_interval(support):
    return support == 'unit_interval'

def is_positive(support):
    return support == 'positive'

def is_simplex(support):
    return support == 'simplex'

def is_real(support):
    return support == 'real'


# TODO: Write a ModelParamList

class ModelParam(abc.ABC):
    def __init__(self, size, support, m=None, s=None):
        if m is None:
            m = torch.randn(size)

        if s is None:
            log_s = torch.randn(size)
        else:
            log_s = s.log()

        self.vp = torch.nn.Parameter(torch.stack([m, log_s]))
        self.size = size
        self.support = support

    def dist(self):
        if self.support in ['simplex', 'unit_interval']:
            return Normal(self.vp[0], self.vp[1].sigmoid()*10)
        else:
            return Normal(self.vp[0], self.vp[1].exp())

    def real_sample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def transform(self, real):
        if is_unit_interval(self.support):
            return real.sigmoid()

        elif is_simplex(self.support):
            return sbt(real)

        elif is_positive(self.support):
            return real.exp()

        elif is_real(self.support):
            return real

        else:
            NotImplemented

    def logabsdetJ(self, real, param):
        if is_unit_interval(self.support):
            logabsdetJ = torch.log(param) + torch.log1p(-param)

        elif is_simplex(self.support):
            logabsdetJ = sbt.log_abs_det_jacobian(real, param)

        elif is_positive(self.support):
            logabsdetJ = real

        elif is_real(self.support):
            logabsdetJ = 0.0

        else:
            NotImplemented

        return logabsdetJ

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

class VI(torch.nn.Module):
    def __init__(self):
        assert(hasattr(self, 'mp'))

        # Call parent's init
        super(VI, self).__init__()

        # Register variational parameters
        for key in self.mp:
            mp = self.mp[key]
            if issubclass(type(mp), ModelParam):
                self.__setattr__(key + '_vp', mp.vp)

    @abc.abstractmethod
    def loglike(self, data, params, misc=None):
        pass

    @abc.abstractmethod
    def log_prior(self, reals, params, misc=None):
        pass

    def log_q(self, reals, misc=None):
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
 
    def forward(self, idx):
        reals = self.sample_reals()
        params = self.transform(reals)
        elbo = self.loglike(idx, params) + self.log_prior(params, reals)
        elbo -= self.log_q(reals)
        return elbo

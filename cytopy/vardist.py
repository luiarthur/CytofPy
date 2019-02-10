import abc
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta
from torch.distributions.log_normal import LogNormal
from torch.nn import Parameter
# from torch.distributions.kl import kl_divergence as kld

# VD: Variational Distribution
# VP: Variational Parameters
# VI: Variational Inference


class VD(abc.ABC):
    @abc.abstractmethod
    def __init__(self, size):
        pass

    @abc.abstractmethod
    def dist(self):
        pass

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_prob(self):
        return self.dist().log_prob()


class VDGamma(VD):
    def __init__(self, size):
        log_conc = torch.randn(size)
        log_rate = torch.randn(size)

        self.vp = Parameter(torch.stack([log_conc, log_rate]), requires_grad=True)
        self.size = size

    def dist(self):
        return Gamma(self.vp[0].exp(), self.vp[1].exp())


class VDLogNormal(VD):
    def __init__(self, size):
        m = torch.randn(size)
        log_s = torch.randn(size)

        self.vp = Parameter(torch.stack([m, log_s]), requires_grad=True)
        self.size = size

    def dist(self):
        return LogNormal(self.vp[0], self.vp[1].exp())


class VDNormal(VD):
    def __init__(self, size):
        m = torch.randn(size)
        log_s = torch.randn(size)

        self.vp = Parameter(torch.stack([m, log_s]), requires_grad=True)
        self.size = size

    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())


class VDBeta(VD):
    def __init__(self, size, log_a_init=None, log_b_init=None):
        if log_a_init is None:
            log_a = torch.randn(size)
        else:
            log_a = torch.zeros(size) + log_a_init

        if log_b_init is None:
            log_b = torch.randn(size)
        else:
            log_b = torch.zeros(size) + log_b_init

        self.vp = Parameter(torch.stack([log_a, log_b]), requires_grad=True)
        self.size = size

    def dist(self):
        return Beta(self.vp[0].exp(), self.vp[1].exp())

class VDDirichlet(VD):
    def __init__(self, size):
        log_conc = torch.randn(size)

        self.vp = Parameter(log_conc, requires_grad=True)
        self.size = size

    def dist(self):
        return Dirichlet(self.vp.exp())

class VI(torch.nn.Module):
    def __init__(self):
        # Call parent's init
        super(VI, self).__init__()

        # Register variational parameters
        self.vd = {}
        for key in self.__dict__:
            param = self.__getattribute__(key)
            if issubclass(type(param), VD):
                self.__setattr__(key + '_vp', param.vp)
                self.vd[key] = param

    @abc.abstractmethod
    def loglike(self, data, params):
        pass

    @abc.abstractmethod
    def kl_qp(self, params):
        pass
 
    def forward(self, data):
        params = self.sample_params()
        elbo = self.loglike(data, params) - self.kl_qp(params)
        return elbo

    def sample_params(self):
        params = {}
        for key in self.vd:
            params[key] = self.vd[key].rsample()
        return params


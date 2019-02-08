import abc
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta
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


class VDNormal(VD):
    def __init__(self, size):
        m = torch.randn(size)
        log_s = torch.randn(size)

        self.vp = Parameter(torch.stack([m, log_s]), requires_grad=True)
        self.size = size

    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())


class VDBeta(VD):
    def __init__(self, size):
        log_a = torch.randn(size)
        log_b = torch.randn(size)

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


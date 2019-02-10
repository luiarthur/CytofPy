import torch
import copy
from torch.distributions import Normal, Gamma, Dirichlet, Beta
from torch.distributions.kl import kl_divergence as kld
from cytopy.vardist import VDGamma, VDNormal, VDBeta, VDDirichlet, VI

from torch.nn import Parameter

def default_dims(data, K:int=30, L=None):
    pass

def default_priors(dims):
    pass

class LocalMod(VI):
    def __init__(self, y, y_init_mean):
        self.y_imp = copy.deepcopy(y)
        self.m = []
        self.I = len(y)
        self.N = [yi.size(0) for yi in y]
        self.J = y[0].size(1)
        
        for i in range(self.I):
            m[i] = torch.isnan(y[i])
            # Check this
            self.y_imp_mean[i][m[i]] = Parameter(torch.randn(m[i].sum()) * y_init_mean)
            self.y_imp_log_s[i][m[i]] = Parameter(torch.zeros(m[i].sum()) - 2)

        # This must be done after assigning variational distributions
        super(Local, self).__init__()


    def loglike(self, data, params):
        pass


    def kl_qp(self, params):
        pass

    # def forward(self, params, idx):
    #     mini_y = [self.y_imp[i][idx[i], :] for i in range(self.I)]
    #     mini_m = [self.m[i][idx[i], :] for i in range(self.I)]

    #     # TODO
    #     for i in range(self.I):
    #         m = self.y_imp_mean[i][m[i]]
    #         s = self.y_imp_log_s[i][m[i]].exp()
    #         mini_y[i][mini_m[i]] = Normal(m, s).rsample()
    #     
    #     elbo = log_joint(mini_y, mini_m, params) - log_q(mini_y, mini_m)
    #     return elbo


import copy

import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta, Bernoulli
from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence as kld
from torch.nn import Parameter

from cytopy.vardist import VDGamma, VDNormal, VDBeta, VDDirichlet, VDLogNormal, VI


class LocalMod(VI):
    def __init__(self, y, y_mean_init, y_sd_init):
        self.I = len(y)
        self.N = [yi.size(0) for yi in y]
        self.J = y[0].size(1)

        self.m = [torch.isnan(yi) for yi in y]
        self.y = [VDNormal(y[i].shape) for i in range(self.I)]
        
        for i in range(self.I):
            # Set means for observed y to observed y
            self.y[i].vp[0].data = y[i].data + 0.0
            # Set means for missing y to y_mean_iniy
            self.y[i].vp[0][self.m[i]] = y_mean_init

            # Set sd for observed y to 0
            self.y[i].vp[1].data = torch.tensor(float('-inf'))
            # Set sd for missing y to y_sd_init
            self.y[i].vp[1][self.m[i]] = torch.log(torch.tensor(y_sd_init))

        # This must be done after assigning variational distributions
        super(LocalMod, self).__init__()

    def loglike(self, g_params, y):
        out = 0.0
        for i in range(self.I):
            logits = -g_params['b0'][i:i+1, :].detach() - g_params['b1'][i:i+1, :].detach() * y[i]
            out += Bernoulli(logits=logits[self.m[i]]).log_prob(1.0).sum()
        return out

    def kl_qp(self, g_params, y):
        # TODO: detach all parameters
        gmod = g_params['gmod']

        out = 0.0
        for i in range(self.I):
            log_p = gmod.loglike(y, g_params)

            vp_m = self.y[i].vp[0][m[i]]
            vp_s = self.y[i].vp[1][m[i]].exp()
            log_q = Normal(vp_m, vp_s).log_prob(y[i][m[i]]).sum()

            out += log_p - log_q

        return out

    def forward(self, g_params):
        y = self.sample_params()
        elbo = self.loglike(g_params, y) - self.kl_qp(g_params, y)
        return elbo

    def sample_params(self):
        return [yi.rsample() for yi in self.y]
        

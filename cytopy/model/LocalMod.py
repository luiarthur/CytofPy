import copy

import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta, Bernoulli
from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence as kld
from torch.nn import Parameter
from cytopy.model.GlobalMod import compute_Z

from cytopy.vardist import VDGamma, VDNormal, VDBeta, VDDirichlet, VDLogNormal, VI


class LocalMod(VI):
    def __init__(self, y, y_mean_init, y_sd_init, tau=0.1):
        self.I = len(y)
        self.N = [yi.size(0) for yi in y]
        self.J = y[0].size(1)
        self.tau = tau

        self.m = [torch.isnan(yi) for yi in y]

        y_m_init = [yi.data for yi in y]
        y_log_s_init = [torch.ones(y[i].shape) * float('-inf') for i in range(self.I)]
        
        for i in range(self.I):
            # Set means for missing y to y_mean_iniy
            y_m_init[i][self.m[i]] = y_mean_init

            # Set sd for missing y to y_sd_init
            y_log_s_init[i][self.m[i]] = torch.log(torch.tensor(y_sd_init))

        self.y = [VDNormal(y[i].shape, y_m_init[i], y_log_s_init[i]) for i in range(self.I)]

        # This must be done after assigning variational distributions
        super(LocalMod, self).__init__()

    # TODO: divide by a constant for larger learning rate?
    def loglike(self, theta, y):
        ll = 0.0
        idx = theta['idx']

        for i in range(self.I):
            mi = self.m[i][idx[i], :]
            yi = y[i][idx[i], :]
            logits = -theta['b0'][i:i+1, :].detach() - theta['b1'][i:i+1, :].detach() * yi
            ll += Bernoulli(logits=logits[mi]).log_prob(1.0).sum()

        return ll

    def kl_qp(self, theta, y):
        idx = theta['idx']

        log_p = 0.0
        log_q = 0.0

        for i in range(self.I):
            # log_p
            mi = self.m[i][idx[i], :]
            yi = y[i][idx[i], :]
            d0 = Normal(-theta['mu0'].cumsum(0)[None, None, :],
                        theta['sig0'][None, None, :]).log_prob(yi[:, :, None])
            d0 += theta['eta0'][i:i+1, :, :].log()
            d0 *= mi[:, :, None].float()

            d1 = Normal(theta['mu1'].cumsum(0)[None, None, :],
                        theta['sig1'][None, None, :]).log_prob(yi[:, :, None])
            d1 += theta['eta1'][i:i+1, :, :].log()
            d1 *= mi[:, :, None].float()
            
            logmix_L0 = torch.logsumexp(d0, 2)
            logmix_L1 = torch.logsumexp(d1, 2)

            b_vec = theta['v'].cumprod(0)
            Z = compute_Z(b_vec[None, :] - Normal(0, 1).cdf(theta['H']), self.tau)
            c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
            d = c.sum(1)

            f = d + theta['W'][i:i+1, :].log()
            log_p += torch.logsumexp(f, 1).sum(0)

            # log_q
            vp_m = self.y[i].vp[0][idx[i], :][mi]
            vp_s = self.y[i].vp[1][idx[i], :][mi].exp()
            log_q += Normal(vp_m, vp_s).log_prob(yi[mi]).sum()

        return log_p - log_q

    def forward(self, theta):
        y = self.sample_params()
        elbo = self.loglike(theta, y) - self.kl_qp(theta, y)
        return elbo

    def sample_params(self):
        return [yi.rsample() for yi in self.y]
        

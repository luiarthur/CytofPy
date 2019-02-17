import torch
from torch.distributions import Normal
from torch.distributions import Categorical

def lam_post(mod):
    lam = []
    for i in range(mod.I):
        W = mod.W.rsample().detach()
        H = mod.H.rsample().detach()
        v = mod.v.rsample().detach()
        Z = (v.cumprod(0) > torch.distributions.Normal(0, 1).cdf(H))
        Z = Z.float()
        eta0 = mod.eta0.rsample().detach()
        eta1 = mod.eta1.rsample().detach()
        mu0 = -mod.delta0.rsample().cumsum(0).detach()
        mu1 =  mod.delta1.rsample().cumsum(0).detach()
        sig0 = mod.sig0.rsample().detach()
        sig1 = mod.sig1.rsample().detach()
        yi = mod.y[i].rsample().detach()


        # compute probs
        d0 = Normal(mu0[None, None, :], sig0[None, None, :]).log_prob(yi[:, :, None])
        d0 += eta0[i:i+1, :, :].log()

        d1 = Normal(mu1[None, None, :], sig1[None, None, :]).log_prob(yi[:, :, None])
        d1 += eta1[i:i+1, :, :].log()

        # Ni x J
        logmix_L0 = torch.logsumexp(d0, 2)
        logmix_L1 = torch.logsumexp(d1, 2)

        c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
        d = c.sum(1)

        f = d + W[i:i+1, :].log()

        # lam_probs = f.exp() / f.exp().sum(1, keepdim=True)
        lam_probs = (f - f.logsumexp(1, keepdim=True)).exp()

        lam.append(Categorical(lam_probs).sample())

    return lam


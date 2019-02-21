import torch
from torch.distributions import Normal
from torch.distributions import Categorical

def sample(mod):
    lam = []

    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)
    mu0 = -params['delta0'].cumsum(0).detach()
    mu1 = params['delta1'].cumsum(0).detach()
    eta0 = params['eta0'].detach()
    eta1 = params['eta1'].detach()
    # sig = params['sig'].detach()
    sig0 = params['sig0'].detach()
    sig1 = params['sig1'].detach()
    H = params['H'].detach()
    v = params['v'].detach()
    # TODO: USE in STICK-BREAKING IBP
    Z = (v.cumprod(0) > Normal(0, 1).cdf(H)).double()
    # Z = (v > Normal(0, 1).cdf(H)).double()
    W = params['W'].detach()

    for i in range(mod.I):
        yi = params['y'][i].detach()
        # compute probs
        d0 = Normal(mu0[None, None, :], sig0[None, None, :]).log_prob(yi[:, :, None])
        # d0 = Normal(mu0[None, None, :], sig[i]).log_prob(yi[:, :, None])
        d0 += eta0[i:i+1, :, :].log()

        d1 = Normal(mu1[None, None, :], sig1[None, None, :]).log_prob(yi[:, :, None])
        # d1 = Normal(mu1[None, None, :], sig[i]).log_prob(yi[:, :, None])
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

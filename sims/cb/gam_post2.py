import torch

from torch.distributions import Normal
from torch.distributions import Categorical

# def sample(mod, lam_draw):
def sample(lam, theta, mod):
    """
    theta, mod: output from sample_params.theta(mod, priors, y)
    lam: draws of lambda
    """
    sd_eps = theta['noisy_sd']
    mu0 = theta['mu0']
    mu1 = theta['mu1']
    eta0 = theta['eta0']
    eta1 = theta['eta1']
    sig = theta['sig']
    Z = theta['Z']

    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)
    mu = torch.cat([mu0, mu1])

    gam = []
    for i in range(mod.I):
        yi = params['y'][i].detach()
        # Ni x J x Lz
        logdmix0 = Normal(mu0[None, None ,:], sig[i]).log_prob(yi[:, :, None])
        logdmix0 += eta0[i][None, :, :].log()
        logdmix1 = Normal(mu1[None, None ,:], sig[i]).log_prob(yi[:, :, None])
        logdmix1 += eta1[i][None, :, :].log()
        Zi = Z[:, lam_draw[i]].transpose(1, 0)[:, :, None]
        numer = torch.cat([(1 - Zi).log() + logdmix0, Zi.log() + logdmix1], dim=-1)

        gam_probs = (numer - numer.logsumexp(-1, keepdim=True)).exp()
        gam_i = Categorical(gam_probs).sample()
        gam.append(gam_i)

    return (gam, mu, sig)


import torch

from torch.distributions import Normal
from torch.distributions import Categorical

def sample(theta, mod):
    """
    theta, mod: output from sample_params.theta(mod, priors, y)
    """
    sd_eps = theta['noisy_sd']
    mu0 = theta['mu0']
    mu1 = theta['mu1']
    eta0 = theta['eta0']
    eta1 = theta['eta1']
    sig = theta['sig']
    Z = theta['Z']

    lam = []
    for i in range(mod.I):
        yi = theta['y'][i]
        eps_i = theta['eps'][i]
        Wi = theta['W'][i]

        # Ni x 1
        log_p0i = eps_i.log() + Normal(0, sd_eps).log_prob(yi).sum(1, keepdim=True)

        # Intermediates
        logdmix0 = Normal(mu0[None, None, :], sig[i]).log_prob(yi[:, :, None])
        logdmix0 += eta0[i:i+1, :, :].log()

        logdmix1 = Normal(mu1[None, None, :], sig[i]).log_prob(yi[:, :, None])
        logdmix1 += eta1[i:i+1, :, :].log()

        # Ni x J
        logmix_L0 = torch.logsumexp(logdmix0, 2)
        logmix_L1 = torch.logsumexp(logdmix1, 2)

        # Ni x J x K
        Z_mix = Z[None, :, :] * logmix_L1[:, :, None] + (1 - Z[None, :, :]) * logmix_L0[:, :, None]

        # Ni x K
        f = Z_mix.sum(1)

        # Ni x K
        log_pki = torch.log1p(-eps_i) + torch.log(Wi[None, :]) + f

        # Ni x (K+1)
        log_pi = torch.cat((log_p0i, log_pki), dim=1)

        # lam probs
        lam_probs = (log_pi - log_pi.logsumexp(1, keepdim=True)).exp()

        lam.append(Categorical(lam_probs).sample())

    return lam

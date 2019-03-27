import torch
from torch.distributions import Normal

# Compute loglikelihood
def sample(mod, y=None):
    # Sample posterior model parameters
    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)
    ll = []

    # Detach model parameters
    mu0 = -params['delta0'].cumsum(0).detach()
    mu1 = params['delta1'].cumsum(0).detach()
    eta0 = params['eta0'].detach()
    eta1 = params['eta1'].detach()
    sig = params['sig2'].detach().sqrt()
    H = params['H'].detach()
    v = params['v'].detach()

    if mod.use_stick_break:
        Z = (v.cumprod(0) > Normal(0, 1).cdf(H)).double()
    else:
        Z = (v > Normal(0, 1).cdf(H)).double()

    W = params['W'].detach()

    for i in range(mod.I):
        if y is None:
            # Used the imputed y[i]
            yi = params['y'][i].detach()
        else:
            # Used the user-provided y[i]
            yi = y[i]

        # compute probs
        d0 = Normal(mu0[None, None, :], sig[i]).log_prob(yi[:, :, None])
        d0 += eta0[i:i+1, :, :].log()

        d1 = Normal(mu1[None, None, :], sig[i]).log_prob(yi[:, :, None])
        d1 += eta1[i:i+1, :, :].log()

        # Ni x J
        logmix_L0 = torch.logsumexp(d0, 2)
        logmix_L1 = torch.logsumexp(d1, 2)

        # Ni x J x K
        c = Z[None, :, :] * logmix_L1[:, :, None] + (1 - Z[None, :, :]) * logmix_L0[:, :, None]

        # Ni x K
        d = c.sum(1)
        # loglike for lam[i]
        lli = d + W[i][None, :].log()

        # TODO: Implement this
        # if mod.model_noisy:
        #     eps_i = params['eps'][i].detach()
        #     lli_quiet = lli + torch.log1p(-eps_i)
        #     lli_noisy = Normal(0, mod.noisy_sd).log_prob(yi).sum(1) + eps_i.log()
        #     # lli = torch.cat([lli_noisy, lli_quiet], dim=-1)
        #     lli = torch.cat([lli_quiet, lli_noisy], dim=-1)
        # else:
        #     ll.append(lli)
        ll.append(lli)

    return ll

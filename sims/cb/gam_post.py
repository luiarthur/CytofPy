import torch
from torch.distributions import Normal
from torch.distributions import Categorical
import loglike
import util

# TODO: TEST
def sample(mod, lam_draw):
    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)
    sig = params['sig2'].sqrt().detach()
    mu0 = -params['delta0'].cumsum(0).detach()
    mu1 = params['delta1'].cumsum(0).detach()
    mu = torch.cat([mu0, mu1])
    eta0 = params['eta0'].detach()
    eta1 = params['eta1'].detach()
    H = params['H'].detach()
    v = params['v'].detach()
    if mod.use_stick_break:
        Z = (v.cumprod(0) > H).double()
    else:
        Z = (v > H).double()

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


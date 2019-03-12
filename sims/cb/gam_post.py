import torch
from torch.distributions import Normal
from torch.distributions import Categorical
import loglike

def sample(mod, lam_draw):
    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)

    gam = []
    for i in range(mod.I):
        yi = params['y'][i].detach()
        mu0 = -params['delta0'].cumsum(0).detach()
        mu1 = -params['delta1'].cumsum(0).detach()
        sig_i = params['sig2'][i].sqrt().detach()
        eta0 = params['eta0'].detach()
        eta1 = params['eta1'].detach()
        H = params['H'].detach()
        v = params['v'].detach()
        if mod.use_stick_break:
            Z = (v.cumprod(0) > Normal(0, 1).cdf(H)).double()
        else:
            Z = (v > Normal(0, 1).cdf(H)).double()

        # Ni x J x Lz
        logdmix0 = Normal(mu0[None, None ,:], sig_i).log_prob(yi[:, :, None])
        logdmix0 += eta0[i][None, :, :].log()
        logdmix1 = Normal(mu1[None, None ,:], sig_i).log_prob(yi[:, :, None])
        logdmix1 += eta1[i][None, :, :].log()
        Zi = Z[:, lam_draw[i]].transpose(1, 0)[:, :, None]
        numer = torch.cat([(1 - Zi) * logdmix0, Zi * logdmix1], dim=-1)

        gam_probs = (numer - numer.logsumexp(-1, keepdim=True)).exp()

        gam_i = Categorical(gam_probs).sample()
        # gam_i[gam_i >= mod.L[0]] -= mod.L[0]
        gam.append(Categorical(gam_probs).sample())

        return gam

# # TEST
# # Note that gam[i][n, j] >= L[0] is for Z[i][n, j] == 1
# lam_draw = [lam[i][0] for i in range(mod.I)]
# gam = sample(mod, lam_draw)

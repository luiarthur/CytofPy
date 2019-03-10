import torch
from torch.distributions import Normal
from torch.distributions import Categorical
import loglike

def sample(mod):
    lam = []
    ll = loglike.sample(mod)

    for i in range(mod.I):
        lli = ll[i]

        # lam_probs = f.exp() / f.exp().sum(1, keepdim=True)
        lam_probs = (lli - lli.logsumexp(1, keepdim=True)).exp()

        lam.append(Categorical(lam_probs).sample())

    return lam

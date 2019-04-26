import torch
import numpy as np
from torch.distributions import Normal, Gamma, Beta, Dirichlet
from torch.distributions.log_normal import LogNormal

def get_one_hot(targets, nb_classes, use_np=False):
    if use_np:
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])
    else:
        # use torch
        res = torch.eye(nb_classes)[targets.clone().reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

def pretty_dist(D, warn=False):
    if isinstance(D, Normal):
        mean = D.loc
        sd = D.scale
        out = 'Normal(mean: {}, sd: {})'.format(mean, sd)
    elif isinstance(D, Gamma):
        conc = D.concentration
        rate = D.rate
        out = 'Gamma(shape: {}, rate: {})'.format(conc, rate)
    elif isinstance(D, LogNormal):
        loc = D.loc
        scale = D.scale
        out = 'LogNormal(loc: {}, scale: {})'.format(loc, scale)
    elif isinstance(D, Beta):
        a = D.concentration1
        b = D.concentration0
        out = 'Beta(a: {}, b: {})'.format(a, b)
    elif isinstance(D, Dirichlet):
        conc = D.concentration
        out = 'Dir(conc: {})'.format(conc)
    else:
        if warn:
            print('WARNING: `pretty_dist` NOT IMPLEMENTED for this distribution!')
        out = str(D)

    return out

import torch
from torch.distributions import Normal

def theta(mod, priors, y=None):
    # Sample posterior model parameters
    idx = [range(mod.N[i]) for i in range(mod.I)]
    params = mod.sample_params(idx)

    # Detach model parameters
    mu0 = -params['delta0'].cumsum(0).detach()
    mu1 = params['delta1'].cumsum(0).detach()
    eta0 = params['eta0'].detach()
    eta1 = params['eta1'].detach()
    sig = params['sig2'].detach().sqrt()
    H = params['H'].detach()
    v = params['v'].detach()
    eps = params['eps'].detach()

    if mod.use_stick_break:
        Z = (v.cumprod(0) > Normal(0, 1).cdf(H)).double()
    else:
        Z = (v > Normal(0, 1).cdf(H)).double()

    W = params['W'].detach()

    out = {}
    out['mu0'] = mu0
    out['mu1'] = mu1
    out['eta0'] = eta0
    out['eta1'] = eta1
    out['sig'] = sig
    out['W'] = W
    out['Z'] = Z
    out['eps'] = eps 
    out['noisy_sd'] = torch.sqrt(torch.tensor(priors['noisy_var']))

    yout = []
    for i in range(mod.I):
        if y is None:
            # Used the imputed y[i]
            yi = params['y'][i].detach()
        else:
            # Used the user-provided y[i]
            yi = y[i]

        yout.append(yi)
    
    out['y'] = yout

    return out, mod

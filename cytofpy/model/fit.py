import torch
import math
import datetime
import numpy as np
from .Model import *
import copy
import sys

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

def check_nan_in_grad(mp, key, fixed_grad, i=None):
    if key == 'y_fc1_weight':
        vp = mp.fc1.weight
    elif key == 'y_fc1_bias':
        vp = mp.fc1.bias
    elif key == 'y_fc2_weight':
        vp = mp.fc2.weight
    elif key == 'y_fc2_bias':
        vp = mp.fc2.bias
    else:
        vp = mp[key].vp

    grad_isnan = torch.isnan(vp.grad)
    if grad_isnan.sum() > 0:
        print("WARNING: Setting a nan gradient to zero in {}!".format(key))
        # print("WARNING: Setting a nan gradient to zero in {}! idx: {}".format(
        #     key, grad_isnan.nonzero()))
        vp.grad[grad_isnan] = 0.0
        fixed_grad[0] = True

def update(opt, mod, idx):
    elbo = mod(idx)
    loss = -elbo
    opt.zero_grad()
    loss.backward()

    fixed_grad = [False]
    with torch.no_grad():
        # check_nan_in_grad(mod.y_vae, 'y_fc1_weight', fixed_grad)
        # check_nan_in_grad(mod.y_vae, 'y_fc1_bias', fixed_grad)
        # check_nan_in_grad(mod.y_vae, 'y_fc2_weight', fixed_grad)
        # check_nan_in_grad(mod.y_vae, 'y_fc2_bias', fixed_grad)
        for key in mod.mp:
            check_nan_in_grad(mod.mp, key, fixed_grad)

    opt.step()
    return elbo, fixed_grad[0]

def fit(y, minibatch_size=500, priors=None, max_iter=1000, lr=1e-1,
        print_freq=10, seed=1, y_mean_init=-6.0, y_sd_init=0.5,
        trace_every=None, eps=1e-6, tau=0.1, backup_every=10,
        y_quantiles=[0, 35, 70], p_bounds=[.05, .8, .05],
        init_mp=None, verbose=1, flush=True):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if trace_every is None:
        if max_iter >= 50:
            trace_every = int(max_iter / 50)
        else:
            trace_every = 1

    m = [torch.isnan(yi) for yi in y]

    model = Model(y=y, m=m, priors=priors, tau=tau, verbose=verbose,
                  y_mean_init=y_mean_init, y_sd_init=y_sd_init)

    if init_mp is not None:
        model.mp = init_mp

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mp = copy.deepcopy(model.mp)

    elbo_hist = []
    trace = []

    elbo_good = True
    for t in range(max_iter + 1):
        # DEBUG
        # if model.Nsum > 10000:
        #     print(model.y_vp[2][:, 4264, :])

        idx = []
        for i in range(model.I):
            if minibatch_size > model.N[i]:
                idx_i = np.arange(model.N[i])
            else:
                idx_i = np.random.choice(model.N[i], minibatch_size, replace=False)
            idx.append(idx_i)

        # Update Model parameters
        elbo, fixed_grad = update(optimizer, model, idx)
        elbo_hist.append(elbo.item())
        elbo_good = not fixed_grad

        if t % print_freq == 0:
            print('{} | iteration: {}/{} | elbo: {}'.format(
                datetime.datetime.now(), t, max_iter, elbo_hist[-1]))

        # if t > 10 and math.isnan(elbo_hist[-1]):
        #     print('nan in elbo. Exiting early.')
        #     break

        if backup_every > 0 and t % backup_every == 0 and elbo_good:
            best_mp = copy.deepcopy(model.mp)

        if trace_every > 0 and t % trace_every == 0 and elbo_good:
            mp = {}
            for key in model.mp:
                if key == 'y':
                    pass
                else:
                    mp[key] = copy.deepcopy(model.mp[key])
            trace.append(mp)

        if t > 10 and abs(elbo_hist[-1] / elbo_hist[-2] - 1) < eps:
            print('Convergence suspected! Ending optimizer early.')
            break

        if flush:
            sys.stdout.flush()

    # FIXME: why can't i return the model?
    #        why can't i return priors?
    #        I could before...
    return {'elbo': elbo_hist, 'trace': trace, 'mp': best_mp, 'tau': tau,
            'y': y, 'priors': str(model.priors)}

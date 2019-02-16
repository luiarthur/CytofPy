import torch
import math
import datetime
import numpy as np
from .Model import *
import copy
import sys

def update(opt, mod, data):
    elbo = mod(data)
    loss = -elbo
    opt.zero_grad()
    loss.backward()
    opt.step()
    return elbo

def fit(y, minibatch_size=500, priors=None, max_iter=1000, lr=1e-1,
        print_freq=10, seed=1, y_mean_init=-6.0, y_sd_init=0.5,
        trace_every=None, eps=1e-6, tau=0.1, save_every=10,
        y_quantiles=[0, 35, 70], p_bounds=[.05, .8, .05],
        verbose=1, flush=True):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if trace_every is None:
        if max_iter >= 50:
            trace_every = int(max_iter / 50)
        else:
            trace_every = 1

    m = [torch.isnan(yi) for yi in y]

    model = Model(y=y, m=m, priors=priors, tau=tau, verbose=verbose)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_model = copy.deepcopy(model)

    elbo_hist = []
    trace = []

    elbo_good = True
    for t in range(max_iter):
        idx = []
        for i in range(model.I):
            idx_i = np.random.choice(model.N[i], minibatch_size)
            idx.append(idx_i)

        # Update Model parameters
        elbo = update(optimizer, model, {'idx': idx})
        elbo_hist.append(elbo.item())

        if t % print_freq == 0:
            print('{} | iteration: {}/{} | elbo: {}'.format(
                datetime.datetime.now(), t, max_iter, elbo_hist[-1]))

        if t > 10 and math.isnan(elbo_hist[-1]):
            print('nan in elbo. Exiting early.')
            break
            # print('mending nan in parameters')
            # for p in model.parameters():
            #     with torch.no_grad():
            #         idx_grad_is_nan = torch.isnan(p.grad)
            #         if idx_grad_is_nan.sum() > 0:
            #             p.grad[idx_grad_is_nan].fill_(0)
            # TODO: use previous best ests

        if save_every > 0 and t % save_every == 0 and elbo_good:
            best_model = copy.deepcopy(model)

        if trace_every > 0 and t % trace_every == 0: # and not repaired_grads:
            trace.append(best_model.vd)

        if t > 10 and abs(elbo_hist[-1] / elbo_hist[-2] - 1) < eps:
            print('Convergence suspected! Ending optimizer early.')
            break

        if flush:
            sys.stdout.flush()

    return {'elbo': elbo_hist, 'model': best_model, 'trace': trace}



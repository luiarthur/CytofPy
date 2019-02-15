import torch
import datetime
import numpy as np
from .GlobalMod import GlobalMod
from .LocalMod import LocalMod
import copy
import sys

def update(opt, mod, dat):
    elbo = mod(dat)
    loss = -elbo
    opt.zero_grad()
    loss.backward()
    opt.step()
    return elbo

def fit(y, minibatch_size=500, priors=None, max_iter=1000, lr_g=1e-1, lr_l=1e-1,
        print_freq=10, seed=1, y_mean_init=-3.0, y_sd_init=0.5,
        trace_g_every=None, trace_l_every=None, eps=1e-6, tau=0.1,
        verbose=1, flush=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if trace_g_every is None:
        if max_iter >= 50:
            trace_g_every = int(max_iter / 50)
        else:
            trace_g_every = 1

    if trace_l_every is None:
        if max_iter >= 10:
            trace_l_every = int(max_iter / 10)
        else:
            trace_l_every = 1


    I = len(y)
    m = [torch.isnan(yi) for yi in y]

    g_model = GlobalMod(priors, tau=tau, verbose=verbose)
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr_g)
    best_g_model = copy.deepcopy(g_model)

    l_model = LocalMod(y, y_mean_init, y_sd_init, tau=tau)
    l_optimizer = torch.optim.Adam([yvd_i.vp for yvd_i in l_model.y], lr=lr_l)
    best_l_model = copy.deepcopy(l_model)

    g_elbo_hist = [float('nan')]
    l_elbo_hist = [float('nan')]
    trace_g = []

    for t in range(max_iter):
        # Global Model
        mini_y = []
        mini_m = []
        idx = []
        
        y_curr = l_model.sample_params()
        for i in range(g_model.I):
            idx_i = np.random.choice(g_model.N[i], minibatch_size)
            idx.append(idx_i)
            # mini_y.append(y[i][idx_i, :]) # Modify this to use the imputed y
            mini_y.append(y_curr[i][idx_i, :].detach())
            mini_m.append(m[i][idx_i, :])

        # Update Global parameters
        g_elbo = update(g_optimizer, g_model, {'y': mini_y, 'm': mini_m, 'isLocal': False})
        g_elbo_hist.append(g_elbo.item())

        # Local Model
        theta_curr = g_model.sample_params()
        for key in theta_curr:
            theta_curr[key] = theta_curr[key].detach()
        theta_curr['idx'] = idx

        # # Update missing values
        l_elbo = update(l_optimizer, l_model, theta_curr)
        l_elbo_hist.append(l_elbo.item())

        if t % print_freq == 0:
            print('{} | iteration: {} / {} | g_elbo: {} | l_elbo: {}'.format(
                datetime.datetime.now(), t, max_iter, g_elbo_hist[-1], l_elbo_hist[-1]))

        if t % trace_g_every == 0: # and not repaired_grads:
            best_g_model = copy.deepcopy(g_model)
            trace_g.append(best_g_model.vd)

        if t > 10 and abs(g_elbo_hist[-1] / g_elbo_hist[-2] - 1) < eps:
            print('Convergence suspected! Ending optimizer early.')
            break

        if flush:
            sys.stdout.flush()

    return {'g_elbo': g_elbo_hist,
            'g_model': best_g_model,
            'trace_g': trace_g,
            'l_elbo': l_elbo_hist}
            # 'l_model': best_l_model}
            # 'l_trace': trace_l}



import torch
import numpy as np
from .GlobalMod import GlobalMod
from .LocalMod import LocalMod
import copy

def fit(y, minibatch_size=500, priors=None, max_iter=1000, lr=1e-1, print_freq=10, seed=1,
        trace_g_every=None, trace_l_every=None, eps=1e-6, iota=1.0, tau=0.1, verbose=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if trace_g_every is None:
        trace_g_every = int(max_iter / 50)

    if trace_l_every is None:
        trace_l_every = int(max_iter / 50)

    I = len(y)
    m = [torch.isnan(yi) for yi in y]

    g_model = GlobalMod(priors, iota=iota, tau=tau, verbose=verbose)
    # l_model = LocalMod()

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    best_g_model = copy.deepcopy(g_model)

    elbo_hist = []
    trace_g = []

    for t in range(max_iter):
        # Global Model
        mini_y = []
        mini_m = []
        
        for i in range(g_model.I):
            idx = np.random.choice(g_model.N[i], minibatch_size)
            mini_y.append(y[i][idx, :])
            mini_m.append(m[i][idx, :])

        elbo = g_model({'y': mini_y, 'm': mini_m})
        loss = -elbo
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()

        elbo_hist.append(-loss.item())
        if t % print_freq == 0:
            print('iteration: {} / {} | elbo: {}'.format(t, max_iter, elbo_hist[-1]))

        if t % trace_g_every == 0: # and not repaired_grads:
            best_g_model = copy.deepcopy(g_model)
            trace_g.append(best_g_model.vd)

        if t > 10 and abs(elbo_hist[-1] / elbo_hist[-2] - 1) < eps:
            print('Convergence suspected! Ending optimizer early.')
            break

    return {'elbo': elbo_hist, 'g_model': best_g_model, 'trace_g': trace_g}



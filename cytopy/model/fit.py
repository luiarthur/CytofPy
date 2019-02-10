import torch
import numpy as np
from .GlobalMod import GlobalMod
from .LocalMod import LocalMod
import copy

def update(opt, mod, dat):
    elbo = mod(dat)
    loss = -elbo
    opt.zero_grad()
    loss.backward()
    opt.step()
    return elbo

def fit(y, minibatch_size=500, priors=None, max_iter=1000, lr=1e-1,
        print_freq=10, seed=1, y_mean_init=-3.0, y_sd_init=0.5,
        trace_g_every=None, trace_l_every=None, eps=1e-6, iota=1.0, tau=0.1,
        verbose=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if trace_g_every is None:
        trace_g_every = int(max_iter / 50)

    if trace_l_every is None:
        trace_l_every = int(max_iter / 10)

    I = len(y)
    m = [torch.isnan(yi) for yi in y]

    g_model = GlobalMod(priors, iota=iota, tau=tau, verbose=verbose)
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    best_g_model = copy.deepcopy(g_model)

    l_model = LocalMod(y, y_mean_init, y_sd_init)
    # TODO: Do these
    # l_optimizer = torch.optim.Adam([yvd_i.vp for yvd_i in l_model.y], lr=lr)
    # best_l_model = copy.deepcopy(l_model)

    g_elbo_hist = []
    trace_g = []

    for t in range(max_iter):
        # Global Model
        mini_y = []
        mini_m = []
        idx = []
        
        for i in range(g_model.I):
            idx_i = np.random.choice(g_model.N[i], minibatch_size)
            idx.append(idx_i)
            mini_y.append(y[i][idx_i, :]) # Modify this to use the imputed y
            mini_m.append(m[i][idx_i, :])

        g_elbo = update(g_optimizer, g_model, {'y': mini_y, 'm': mini_m, 'isLocal': False})
        # g_params = g_model.parameters.data
        # g_params['isLocal'] = True
        # g_params['gmod'] = g_model
        # l_elbo = update(l_optimizer, l_model, g_params)

        g_elbo_hist.append(g_elbo.item())
        # l_elbo_hist.append(g_elbo.item())

        if t % print_freq == 0:
            print('iteration: {} / {} | g_elbo: {} | l_elbo: {}'.format(t, max_iter, g_elbo_hist[-1], 0.0))

        if t % trace_g_every == 0: # and not repaired_grads:
            best_g_model = copy.deepcopy(g_model)
            trace_g.append(best_g_model.vd)

        if t > 10 and abs(g_elbo_hist[-1] / g_elbo_hist[-2] - 1) < eps:
            print('Convergence suspected! Ending optimizer early.')
            break

    return {'g_elbo': g_elbo_hist,
            'g_model': best_g_model,
            'trace_g': trace_g,
            # 'l_elbo': l_elbo_hist,
            # 'l_model': best_l_model,,
            # 'l_elbo': trace_l
            }



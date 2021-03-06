import os
import torch

import cytofpy

import math
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle


def add_gridlines_Z(Z):
    J, K = Z.shape
    for j in range(J):
        plt.axhline(y=j+.5, color='grey', linewidth=.5)

    for k in range(K):
        plt.axvline(x=k+.5, color='grey', linewidth=.5)


if __name__ == '__main__':
    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)

    show_plots = True

    # torch.manual_seed(2) # Data with seed(2) is good
    torch.manual_seed(0) # This data is good
    np.random.seed(0)

    # Success
    # data = cytofpy.util.simdata(N=[30000, 10000, 20000], L0=1, L1=1, J=4, a_W=[300, 700])
    # data = cytofpy.util.simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=4, a_W=[300, 700])

    # TODO: Make this work
    data = cytofpy.util.simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=8)

    cb = data['data']
    y = copy.deepcopy(cb['y'])
    I = len(y)

    # Color map
    cm_greys = plt.cm.get_cmap('Greys')
    cm = plt.cm.get_cmap('bwr')
    cm.set_under(color='blue')
    cm.set_over(color='red')
    cm.set_bad(color='black')


    if show_plots:
        plt.imshow(data['params']['Z'], aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        J, K = data['params']['Z'].shape
        add_gridlines_Z(data['params']['Z'])
        plt.savefig('{}/Z_true.pdf'.format(path_to_exp_results))
        plt.show()

        # Plot yi histograms
        plt.hist(y[0][:, 1], bins=100, density=True); plt.xlim(-15, 15); plt.show()
        plt.hist(y[1][:, 3], bins=100, density=True); plt.xlim(-15, 15); plt.show()
        plt.hist(y[2][:, -1], bins=100, density=True); plt.xlim(-15, 15); plt.show()

        # Heatmaps
        for i in range(I):
            plt.imshow(y[i], aspect='auto', vmin=-2, vmax=2, cmap=cm)
            plt.colorbar()
            plt.show()

    K = 10
    L = [2, 2]

    # model.debug=True
    priors = cytofpy.model.default_priors(y, K=K, L=L)
    out = cytofpy.model.fit(y, max_iter=5000, lr=1e-1, print_freq=1, eps_conv=1e-6,
                            priors=priors, minibatch_size=100000, tau=0.1,
                            verbose=0, seed=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))

    elbo = out['elbo']

    out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))
    mod = cytofpy.model.Model(y=out['y'], priors=priors,
                              tau=out['tau'],
                              use_stick_break=out['use_stick_break'],
                              model_noisy=out['model_noisy'])


    plt.plot(elbo)
    plt.ylabel('ELBO / NSUM')
    plt.show()

    # Posterior Inference
    B = 100
    idx = [np.random.choice(mod.N[i], 1) for i in range(mod.I)]
    post = [mod.sample_params(idx) for b in range(B)]

    # Plot Z
    H = torch.stack([p['H'] for p in post]).detach().reshape((B, mod.J, mod.K))
    v = torch.stack([p['v'] for p in post]).detach().reshape((B, 1, mod.K))
    Z = (v.cumprod(2) > H).numpy()
    plt.imshow(Z.mean(0) > .5, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    add_gridlines_Z(Z[0])
    plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
    plt.show()


    # Plot sig
    # sig0 = torch.stack([p['sig0'] for p in post]).detach().numpy()
    # plt.boxplot(sig0, showmeans=True, whis=[2.5, 97.5], showfliers=False)
    # plt.xlabel('$\sigma$_0', fontsize=15)
    # for yint in data['params']['sig'].tolist():
    #     plt.axhline(yint)

    # plt.show()

    W = torch.stack([p['W'] for p in post]).detach().numpy()
    v = torch.stack([p['v'] for p in post]).detach().numpy()
    alpha = torch.stack([p['alpha'] for p in post]).detach().numpy()

    # Plot W, v
    plt.figure()
    for i in range(mod.I):
        plt.subplot(mod.I + 1, 1, i + 1)
        plt.boxplot(W[:, i, :], showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)
        for yint in data['params']['W'][i, :].tolist():
            plt.axhline(yint)

    plt.subplot(mod.I + 1, 1, mod.I + 1)
    plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
    plt.ylabel('$v$', rotation=0, labelpad=15)
    plt.tight_layout()
    plt.show()


    # Trace plots of variational parameters
    W_trace = torch.stack([t['W'].dist().mean for t in out['trace_g']]).detach().numpy()
    v_trace = torch.stack([t['v'].dist().mean for t in out['trace_g']]).detach().numpy()

    # Trace for v
    plt.plot(v_trace)
    plt.title('v trace')
    plt.show()

    # Plot sig mean trace
    # sig0_m_trace = torch.stack([t['sig0'].dist().mean for t in out['trace_g']])
    # plt.plot(sig0_m_trace.detach().numpy())

    # for i in range(mod.I):
    #     plt.axhline(data['params']['sig'][i])

    # plt.title('trace plot for $\sigma$_0 vp mean')
    # plt.show()




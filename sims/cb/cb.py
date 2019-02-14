import os
import torch

import cytopy

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
    torch.set_num_threads(4)

    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)
    path_to_cb_data = 'data/cb.txt'

    show_plots = False

    # torch.manual_seed(2) # Data with seed(2) is good
    torch.manual_seed(0) # This data is good
    np.random.seed(0)

    # Success
    data = cytopy.util.readCB(path_to_cb_data)
    y = copy.deepcopy(data['y'])
    I = len(y)

    # Color map
    cm_greys = plt.cm.get_cmap('Greys')
    cm = plt.cm.get_cmap('bwr')
    cm.set_under(color='blue')
    cm.set_over(color='red')
    cm.set_bad(color='black')

    if show_plots:
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
    priors = cytopy.model.default_priors(y, K=K, L=L)
    out = cytopy.model.fit(y, max_iter=5000, lr_g=1e-1, lr_l=1e-2, print_freq=10, eps=1e-6,
                           priors=priors, minibatch_size=100, tau=0.1,
                           verbose=0, seed=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))
    elbo = out['g_elbo']

    if show_plots:
        out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))
        gmod = out['g_model']

        plt.plot(elbo)
        plt.ylabel('ELBO / NSUM')
        plt.show()

        # Posterior Inference
        B = 100
        post = [gmod.sample_params() for b in range(B)]

        # Plot Z
        H = torch.stack([p['H'] for p in post]).detach().reshape((B, gmod.J, gmod.K))
        v = torch.stack([p['v'] for p in post]).detach().reshape((B, 1, gmod.K))
        Z = (v.cumprod(2) > torch.distributions.Normal(0, 1).cdf(H)).numpy()
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
        for i in range(gmod.I):
            plt.subplot(gmod.I + 1, 1, i + 1)
            plt.boxplot(W[:, i, :], showmeans=True, whis=[2.5, 97.5], showfliers=False)
            plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)
            for yint in data['params']['W'][i, :].tolist():
                plt.axhline(yint)

        plt.subplot(gmod.I + 1, 1, gmod.I + 1)
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

        # for i in range(gmod.I):
        #     plt.axhline(data['params']['sig'][i])

        # plt.title('trace plot for $\sigma$_0 vp mean')
        # plt.show()




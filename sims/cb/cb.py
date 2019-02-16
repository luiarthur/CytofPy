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
    torch.set_num_threads(8)

    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)
    path_to_cb_data = 'data/cb.txt'

    show_plots = False

    # torch.manual_seed(2) # Data with seed(2) is good
    torch.manual_seed(0) # This data is good
    np.random.seed(0)

    # Read Data
    data = cytopy.util.readCB(path_to_cb_data)
    cytopy.util.preprocess(data, rm_cells_below=-6.0)
    print(data['N'])

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
        plt.savefig('{}/y{}.pdf'.format(path_to_exp_results, i))
        if show_plots:
            plt.show()

    K = 30
    L = [5, 5]

    # model.debug=True
    priors = cytopy.model.default_priors(y, K=K, L=L)
    out = cytopy.model.fit(y, max_iter=1000, lr=1e-1, print_freq=10, eps=1e-6,
                           priors=priors, minibatch_size=1000, tau=0.1,
                           trace_every=0, save_every=10,
                           verbose=2, seed=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))
    elbo = out['elbo']
    mod = out['model']

    if show_plots:
        # out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

        plt.plot(elbo)
        plt.ylabel('ELBO / NSUM')
        plt.show()

        # Posterior Inference
        B = 100
        idx = [np.random.choice(mod.N[i], 100) for i in range(mod.I)]
        post = [mod.sample_params(idx) for b in range(B)]

        # Plot Z
        H = torch.stack([p['H'] for p in post]).detach().reshape((B, mod.J, mod.K))
        v = torch.stack([p['v'] for p in post]).detach().reshape((B, 1, mod.K))
        Z = (v.cumprod(2) > torch.distributions.Normal(0, 1).cdf(H)).numpy()
        plt.imshow(Z.mean(0) > .5, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        add_gridlines_Z(Z[0])
        plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
        plt.show()


        # Plot mu
        mu0 = -torch.stack([p['delta0'] for p in post]).detach().cumsum(1).numpy()
        mu1 = torch.stack([p['delta1'] for p in post]).detach().cumsum(1).numpy()
        mu = np.concatenate([mu0[:, ::-1], mu1], axis=1)
        plt.boxplot(mu, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$\mu$', rotation=0, labelpad=15)
        plt.axhline(0)
        plt.axvline(mu0.shape[1] + .5)
        plt.show()

        # y0
        # FIXME: the observed y's are being changed!
        y0 = torch.stack([mod.y[0].rsample() for b in range(10)])
        y0.mean(0)
        y[0]

        # TODO: check sig0, sig1

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

        plt.subplot(mod.I + 1, 1, mod.I + 1)
        plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$v$', rotation=0, labelpad=15)
        plt.tight_layout()
        plt.show()


        # TODO: Plot b0, b1
        ygrid = torch.arange(-8, 8, .1)
        pm = cytopy.model.prob_miss(ygrid[:, None, None],
                                    mod.b0[None, :, :], mod.b1[None, :, :], mod.b2[None, :, :])
        # Plot prob miss for i=0, j=2
        plt.plot(ygrid.numpy(), pm[:, 0, 6].numpy()); plt.show()

        # Trace plots of variational parameters
        W_trace = torch.stack([t['W'].dist().mean for t in out['trace']]).detach().numpy()
        v_trace = torch.stack([t['v'].dist().mean for t in out['trace']]).detach().numpy()

        # Trace for v
        plt.plot(v_trace)
        plt.title('v trace')
        plt.show()

        # Plot sig mean trace
        # sig0_m_trace = torch.stack([t['sig0'].dist().mean for t in out['trace']])
        # plt.plot(sig0_m_trace.detach().numpy())

        # for i in range(mod.I):
        #     plt.axhline(data['params']['sig'][i])

        # plt.title('trace plot for $\sigma$_0 vp mean')
        # plt.show()




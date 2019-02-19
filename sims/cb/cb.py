import os
import torch
from torch.distributions.log_normal import LogNormal
from torch.distributions import Gamma

import cytopy
from lam_post import lam_post

import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
import numpy as np
import pickle

# Use smaller learning rate, with double precision
# https://discuss.pytorch.org/t/why-double-precision-training-sometimes-performs-much-better/31194

def relabel_lam(lami_est, wi_mean):
    K = wi_mean.shape[0]
    k_ord = wi_mean.argsort()
    lami_new = lami_est + 0
    counts = []
    for k in range(K):
        idx_k = lami_est == k_ord[k]
        lami_new[idx_k] = k
        counts.append(idx_k.sum())
    return (lami_new, counts)


def add_gridlines_Z(Z):
    J, K = Z.shape
    for j in range(J):
        plt.axhline(y=j+.5, color='grey', linewidth=.5)

    for k in range(K):
        plt.axvline(x=k+.5, color='grey', linewidth=.5)


if __name__ == '__main__':
    torch.set_num_threads(0)

    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)
    os.makedirs('{}/pm/'.format(path_to_exp_results), exist_ok=True)
    path_to_cb_data = 'data/cb.txt'

    show_plots = False

    # torch.manual_seed(2) # Data with seed(2) is good
    torch.manual_seed(0) # This data is good
    np.random.seed(0)

    # Read Data
    data = cytopy.util.readCB(path_to_cb_data)
    I = len(data['y'])

    # remove markers that are highly missing/negative or positive
    good_markers = [True, False, True, False, True, False, True, True, False,
                    True, False, True, True, True, False, True, True, False, False,
                    True, False, True, True, True, True, False, True, False, True,
                    True, False, True]
    good_markers = torch.tensor(good_markers).nonzero().squeeze()

    for i in range(I):
        data['y'][i] = data['y'][i][:, good_markers]
        data['m'][i] = data['m'][i][:, good_markers]

    # remove cells with expressions below -6
    cytopy.util.preprocess(data, rm_cells_below=-6.0)
    print(data['N'])
    y = copy.deepcopy(data['y'])

    # Color map
    cm_greys = plt.cm.get_cmap('Greys', 5)
    VMIN, VMAX = (-3, 3) 
    cm = plt.cm.get_cmap('bwr', 7)
    cm.set_under(color='blue')
    cm.set_over(color='red')
    cm.set_bad(color='black')

    # Plot yi histograms
    # plt.hist(y[0][:, 1], bins=100, density=True); plt.xlim(-15, 15); plt.show()
    # plt.hist(y[1][:, 3], bins=100, density=True); plt.xlim(-15, 15); plt.show()
    # plt.hist(y[2][:, -1], bins=100, density=True); plt.xlim(-15, 15); plt.show()

    # Heatmaps
    for i in range(I):
        plt.imshow(y[i], aspect='auto', vmin=VMIN, vmax=VMAX, cmap=cm)
        plt.colorbar()
        plt.savefig('{}/y{}.pdf'.format(path_to_exp_results, i + 1))
        plt.close()

    K = 10
    L = [5, 5]

    # model.debug=True
    y_bounds = [-6., -4.5, -3.]
    priors = cytopy.model.default_priors(y, K=K, L=L,
                                         y_bounds=y_bounds, p_bounds=[.01, .8, .01])
                                         # y_quantiles=[0, 25, 50], p_bounds=[.01, .8, .01])
                                         # y_quantiles=[1, 5, 10], p_bounds=[.05, .8, .05])
    priors['sig'] = LogNormal(-1, .01)
    priors['alpha'] = Gamma(.1, .1)

    # Missing Mechanism
    ygrid = torch.arange(-8, 8, .1)
    pm = cytopy.model.prob_miss(ygrid[:, None, None],
                                priors['b0'][None, :, :],
                                priors['b1'][None, :, :],
                                priors['b2'][None, :, :])

    # Plot prob miss for each (i, j)
    for i in range(priors['I']):
        for j in range(priors['J']):
            plt.plot(ygrid.numpy(), pm[:, i, j].numpy())
            plt.savefig('{}/pm/pm_i{}_j{}.pdf'.format(path_to_exp_results, i+1, j+1))
            plt.close()

    out = cytopy.model.fit(y, max_iter=50, lr=1e-1, print_freq=10, eps=1e-6,
                           y_mean_init=y_bounds[1], y_sd_init=0.1,
                           priors=priors, minibatch_size=1000, tau=0.1,
                           trace_every=50, backup_every=10,
                           verbose=0, seed=1)

    out = out['model']
    out = cytopy.model.fit(y, max_iter=10000, lr=1e-2, print_freq=10, eps=1e-6,
                           y_mean_init=y_bounds[1], y_sd_init=0.1,
                           priors=priors, minibatch_size=1000, tau=0.1,
                           trace_every=50, backup_every=10,
                           init=out, verbose=0, seed=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))
    elbo = out['elbo']
    mod = out['model']

    show_plots = True
    if show_plots:
        print("Making Plots...")

        # out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))
        plt.show()

        plt.plot(elbo)
        plt.ylabel('ELBO / NSUM')
        plt.savefig('{}/elbo.pdf'.format(path_to_exp_results))
        plt.close()

        # Posterior Inference
        B = 1000
        idx = [np.random.choice(mod.N[i], 1) for i in range(mod.I)]
        post = [mod.sample_params(idx) for b in range(B)]

        # Plot Z
        H = torch.stack([p['H'] for p in post]).detach().reshape((B, mod.J, mod.K))
        v = torch.stack([p['v'] for p in post]).detach().reshape((B, 1, mod.K))
        Z = (v.cumprod(2) > torch.distributions.Normal(0, 1).cdf(H)).numpy()
        plt.imshow(Z.mean(0), aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        add_gridlines_Z(Z[0])
        plt.colorbar()
        plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
        plt.close()


        # Plot mu
        mu0 = -torch.stack([p['delta0'] for p in post]).detach().cumsum(1).numpy()
        mu1 = torch.stack([p['delta1'] for p in post]).detach().cumsum(1).numpy()
        mu = np.concatenate([mu0[:, ::-1], mu1], axis=1)
        plt.boxplot(mu, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$\mu$', rotation=0, labelpad=15)
        plt.axhline(0)
        plt.axvline(mu0.shape[1] + .5)
        plt.savefig('{}/mu.pdf'.format(path_to_exp_results))
        plt.close()

        # y0
        # FIXME: the observed y's are being changed!
        for i in range(mod.I):
            yi = torch.stack([mod.y[i].rsample().detach() for b in range(10)])
            plt.hist(yi.mean(0)[mod.m[i]].numpy())
            plt.savefig('{}/y{}_imputed_hist.pdf'.format(path_to_exp_results, i + 1))
            plt.close()

        # Plot sig
        sig = torch.stack([p['sig'] for p in post]).detach().numpy()
        plt.boxplot(sig, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.xlabel('$\sigma$', fontsize=15)
        plt.savefig('{}/sig.pdf'.format(path_to_exp_results))
        plt.close()

        # sig0 = torch.stack([p['sig0'] for p in post]).detach().numpy()
        # plt.boxplot(sig0[:, ::-1], showmeans=True, whis=[2.5, 97.5], showfliers=False)
        # plt.xlabel('$\sigma$0', fontsize=15)
        # plt.savefig('{}/sig0.pdf'.format(path_to_exp_results))
        # plt.close()

        # sig1 = torch.stack([p['sig1'] for p in post]).detach().numpy()
        # plt.boxplot(sig1, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        # plt.xlabel('$\sigma$1', fontsize=15)
        # plt.savefig('{}/sig1.pdf'.format(path_to_exp_results))
        # plt.close()

        # Plot W, v
        W = torch.stack([p['W'] for p in post]).detach().numpy()
        v = torch.stack([p['v'] for p in post]).detach().numpy()
        alpha = torch.stack([p['alpha'] for p in post]).detach().numpy()

        plt.figure()
        for i in range(mod.I):
            plt.subplot(mod.I + 1, 1, i + 1)
            plt.boxplot(W[:, i, :], showmeans=True, whis=[2.5, 97.5], showfliers=False)
            plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)

        plt.subplot(mod.I + 1, 1, mod.I + 1)
        plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$v$', rotation=0, labelpad=15)
        plt.tight_layout()
        plt.savefig('{}/W_v.pdf'.format(path_to_exp_results))
        plt.close()


        # Trace plots of variational parameters
        W_trace = torch.stack([t['W'].dist().mean for t in out['trace']]).detach().numpy()
        v_trace = torch.stack([t['v'].dist().mean for t in out['trace']]).detach().numpy()

        for i in range(mod.I):
            plt.plot(W_trace[:, i, :])
            plt.title('w trace i={}'.format(i+1))
            plt.savefig('{}/W{}_trace.pdf'.format(path_to_exp_results, i+1))
            plt.close()

        # Trace for v
        plt.plot(v_trace)
        plt.title('v trace')
        plt.savefig('{}/v_trace.pdf'.format(path_to_exp_results))
        plt.close()

        # Plot sig mean trace
        sig_trace = torch.stack([t['sig'].dist().mean for t in out['trace']])
        plt.plot(sig_trace.detach().numpy()[2:])
        plt.title('sig trace')
        plt.savefig('{}/sig_trace.pdf'.format(path_to_exp_results))
        plt.close()

        # sig0_trace = torch.stack([t['sig0'].dist().mean for t in out['trace']])
        # plt.plot(sig0_trace.detach().numpy()[2:])
        # plt.title('sig0 trace')
        # plt.savefig('{}/sig0_trace.pdf'.format(path_to_exp_results))
        # plt.close()

        # sig1_trace = torch.stack([t['sig1'].dist().mean for t in out['trace']])
        # plt.plot(sig1_trace.detach().numpy()[2:])
        # plt.title('sig1 trace')
        # plt.savefig('{}/sig1_trace.pdf'.format(path_to_exp_results))
        # plt.close()

        # lam posterior
        # TODO: This is just a simple version
        lam = [lam_post(mod) for b in range(30)]
        lam = [torch.stack([lam_b[i] for lam_b in lam]) for i in range(mod.I)]
        lam_est = [lam_i.mode(0)[0] for lam_i in lam]

        W_mean = W.mean(0)
        Z_mean = Z.mean(0)

        for i in range(I):
            k_ord = W_mean[i, :].argsort()
            z_cols = []
            for k in k_ord.tolist():
                if Z_mean[:, k].sum() > 0:
                    z_cols.append(k)
            z_cols = np.array(z_cols)
            Z_hat = Z_mean[:, z_cols]
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5]) 
            plt.subplot(gs[0])
            plt.imshow(Z_hat, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
            plt.colorbar()
            plt.xticks(np.arange(len(z_cols)), z_cols + 1)
            plt.yticks(np.arange(mod.J), np.arange(mod.J) + 1)
            add_gridlines_Z(Z_hat)
            lami_new, counts = relabel_lam(lam_est[i], W_mean[i, :])
            counts_cumsum = np.cumsum(counts)
            plt.subplot(gs[1])
            yi = y[i][lami_new.argsort(), :].numpy().T
            plt.imshow(yi, aspect='auto', vmin=VMIN, vmax=VMAX, cmap=cm)
            for c in counts_cumsum[:-1]:
                plt.axvline(c, color='yellow')
            plt.colorbar()
            plt.yticks([])
            plt.tight_layout()
            plt.savefig('{}/y{}_post.pdf'.format(path_to_exp_results, i + 1))
            plt.close()


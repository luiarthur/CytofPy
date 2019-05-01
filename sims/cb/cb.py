import sys
import os
import util

import torch
from torch.distributions.log_normal import LogNormal
from torch.distributions import Gamma, Dirichlet, Beta, Normal

import cytofpy
import gam_post
from plot_yz import add_gridlines_Z, plot_yz
import Timer

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib 
matplotlib.use('Agg')

import math
import copy
import numpy as np
import pickle
import seaborn as sns
import ddenExpressed
import blue2red

from cytofpy.model import post_proc

# Use smaller learning rate, with double precision
# https://discuss.pytorch.org/t/why-double-precision-training-sometimes-performs-much-better/31194

if __name__ == '__main__':
    torch.set_num_threads(1)

    if len(sys.argv) > 1:
        path_to_exp_results = sys.argv[1]
        SEED = int(sys.argv[2])
    else:
        path_to_exp_results = 'results/sim1-vae/test/'
        SEED = 0

    subsample = 1.0
    # subsample = 0.1

    img_dir = path_to_exp_results + '/img/'
    os.makedirs('{}/dden-expressed'.format(img_dir), exist_ok=True)
    path_to_cb_data = 'data/cb.txt'

    show_plots = False

    # torch.manual_seed(2) # Data with seed(2) is good
    torch.manual_seed(0) # This data is good
    np.random.seed(0)

    # Read Data
    data = cytofpy.util.readCB(path_to_cb_data)
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
    cytofpy.util.preprocess(data, rm_cells_below=-6.0)
    y = copy.deepcopy(data['y'])

    # Get a subsampmle of data
    if 0 < subsample < 1:
        for i in range(len(y)):
            Ni = y[i].shape[0]
            idx = np.random.choice(Ni, int(Ni * subsample), replace=False)
            y[i] = y[i][idx, :]

    # Print size of data
    print('N: {}'.format([yi.shape[0] for yi in y]))

    # Color map
    cm_greys = plt.cm.get_cmap('Greys', 5)
    VMIN, VMAX = VLIM = (-4, 4) 
    cm = blue2red.cm(9)

    # Plot yi histograms
    # plt.hist(y[0][:, 1], bins=100, density=True); plt.xlim(-15, 15); plt.show()
    # plt.hist(y[1][:, 3], bins=100, density=True); plt.xlim(-15, 15); plt.show()
    # plt.hist(y[2][:, -1], bins=100, density=True); plt.xlim(-15, 15); plt.show()

    # Heatmaps
    for i in range(I):
        plt.imshow(y[i], aspect='auto', vmin=VMIN, vmax=VMAX, cmap=cm)
        plt.colorbar()
        plt.savefig('{}/y{}.pdf'.format(img_dir, i + 1))
        plt.close()

    K = 30
    L = [5, 3]

    # model.debug=True

    # y_quantiles=[0, 5, 15]; p_bounds=[.05, .8, .05]
    # y_quantiles=[0, 35, 70]; p_bounds=[.05, .8, .05]
    # y_quantiles=[0, 35, 70]; p_bounds=[.01, .8, .01] # BAD
    # y_quantiles=[0, 25, 50]; p_bounds=[.01, .8, .01] # BAD
    # y_quantiles=[30, 50, 70]; p_bounds=[.01, .8, .01] # Good

    # y_quantiles=[40, 50, 60]; p_bounds=[.01, .8, .01] # BEST
    y_quantiles=[0, 25, 50]; p_bounds=[.05, .8, .05]

    priors = cytofpy.model.default_priors(y, K=K, L=L, y_quantiles=y_quantiles, p_bounds=p_bounds)

    priors['sig2'] = Gamma(.1, 1)
    priors['alpha'] = Gamma(.1, .1)
    # priors['alpha'] = Gamma(2, .1)
    priors['delta0'] = Gamma(1, 1)
    priors['delta1'] = Gamma(1, 1)
    priors['noisy_var'] = 10.0
    priors['eps'] = Beta(1, 99)

    # Missing Mechanism
    ygrid = torch.arange(-8, 1, .1)
    pm = cytofpy.model.prob_miss(ygrid[:, None],
                                 priors['b0'][None, :],
                                 priors['b1'][None, :],
                                 priors['b2'][None, :])

    y_peak = ygrid[pm[:, 0].argmax()]
    y_bounds = [None, y_peak, None]
    print('y peak: {}'.format(y_peak))

    # Plot prob miss for each (i, j)
    plt.figure()
    for i in range(priors['I']):
        plt.plot(ygrid.numpy(), pm[:, i].numpy(), label='i: {}'.format(i + 1))
        plt.ylabel('prob. of missing')
        plt.xlabel('y')

    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/prob_miss.pdf'.format(img_dir, i+1))
    plt.close()

    print('\nPriors & Constants:', flush=True)
    print('y_quantiles: {}'.format(y_quantiles), flush=True)
    print('p_bounds: {}'.format(p_bounds), flush=True)
    for key in priors:
        print('{}: {}'.format(key, util.pretty_dist(priors[key])), flush=True)

    # TODO: max_iter = 10000, minibatch_size=2000
    with Timer.Timer('Model training'):
        out = cytofpy.model.fit(y, max_iter=10000, lr=1e-2, print_freq=10, eps_conv=0,
                                priors=priors, minibatch_size=2000, tau=0.001,
                                trace_every=50, backup_every=50,
                                verbose=0, seed=SEED, use_stick_break=False)
    # Flush output
    sys.stdout.flush()

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))
    show_plots = True
    if show_plots:
        print("Making Plots...", flush=True)

        # out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

        elbo = out['elbo']
        use_stick_break = out['use_stick_break']
        mod = cytofpy.model.Model(y=y, priors=priors,
                                  tau=out['tau'],
                                  use_stick_break=use_stick_break,
                                  model_noisy=out['model_noisy'])
        mod.mp = out['mp']
        vae = out['vae']
        mod.y_vae =vae
        model_noisy = out['model_noisy']

        plt.plot(elbo)
        plt.ylabel('ELBO / NSUM')
        plt.xlabel('iteration')
        plt.savefig('{}/elbo.pdf'.format(img_dir))
        plt.close()

        tail = 1000
        plt.plot(elbo[-tail:])
        plt.ylabel('ELBO / NSUM')
        plt.xlabel('iteration')
        plt.savefig('{}/elbo_tail.pdf'.format(img_dir))
        plt.close()

        # Posterior Inference
        B = 100
        idx = [np.random.choice(mod.N[i], 1) for i in range(mod.I)]
        post = [mod.sample_params(idx) for b in range(B)]

        # Plot Z
        H = torch.stack([p['H'] for p in post]).detach().reshape((B, mod.J, mod.K))
        v = torch.stack([p['v'] for p in post]).detach().reshape((B, 1, mod.K))
        if use_stick_break:
            # Z = (v.cumprod(2) > torch.distributions.Normal(0, 1).cdf(H)).numpy()
            Z = (v.cumprod(2) > H).numpy()
        else:
            # Z = (v > torch.distributions.Normal(0, 1).cdf(H)).numpy()
            Z = (v > H).numpy()
        plt.imshow(Z.mean(0), aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        add_gridlines_Z(Z[0])
        plt.yticks(np.arange(mod.J), np.arange(mod.J) + 1, fontsize=10)
        plt.xticks(np.arange(mod.K), np.arange(mod.K) + 1, fontsize=10, rotation=90)
        plt.colorbar()
        plt.savefig('{}/Z.pdf'.format(img_dir))
        plt.close()


        # Plot mu
        mu0 = -torch.stack([p['delta0'] for p in post]).detach().cumsum(1).numpy()
        mu1 = torch.stack([p['delta1'] for p in post]).detach().cumsum(1).numpy()
        mu = np.concatenate([mu0[:, ::-1], mu1], axis=1)
        plt.boxplot(mu, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$\mu$', rotation=0, labelpad=15)
        plt.axhline(0)
        plt.axvline(mu0.shape[1] + .5)
        plt.savefig('{}/mu.pdf'.format(img_dir))
        plt.close()

        # eta1
        eta1 = torch.stack([p['eta1'] for p in post]).detach().numpy()
        eta1_mean = eta1.mean(0)

        # Plot eps
        if model_noisy:
            eps = torch.stack([p['eps'] for p in post]).detach().numpy()
            plt.boxplot(eps, showmeans=True, whis=[2.5, 97.5], showfliers=False)
            plt.xlabel('eps', fontsize=15)
            plt.savefig('{}/eps.pdf'.format(img_dir))
            plt.close()


        # Plot sig
        sig2 = torch.stack([p['sig2'] for p in post]).detach().numpy()
        plt.boxplot(sig2, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.xlabel('$\sigma^2$', fontsize=15)
        plt.savefig('{}/sig2.pdf'.format(img_dir))
        plt.close()

        # Plot W, v
        W = torch.stack([p['W'] for p in post]).detach().numpy()
        v = torch.stack([p['v'] for p in post]).detach().numpy()

        plt.figure()
        for i in range(mod.I):
            plt.subplot(mod.I + 1, 1, i + 1)
            plt.boxplot(W[:, i, :], showmeans=True, whis=[2.5, 97.5], showfliers=False)
            plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)

        plt.subplot(mod.I + 1, 1, mod.I + 1)
        if use_stick_break:
            plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
        else:
            plt.boxplot(v, showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$v$', rotation=0, labelpad=15)
        plt.tight_layout()
        plt.savefig('{}/W_v.pdf'.format(img_dir))
        plt.close()

        # Plot alpha
        alpha = torch.stack([p['alpha'] for p in post]).detach().numpy()
        plt.hist(alpha, density=True)
        plt.ylabel('density')
        plt.xlabel('alpha')
        plt.axvline(alpha.mean(0), color='black', linestyle='--')
        plt.axvline(np.percentile(alpha, 2.5), color='black', linestyle='--')
        plt.axvline(np.percentile(alpha, 97.5), color='black', linestyle='--')
        plt.savefig('{}/alpha.pdf'.format(img_dir))
        plt.close()


        # Trace plots of variational parameters
        W_trace = torch.stack([t['W'].dist().mean for t in out['trace']]).detach().numpy()
        v_trace = torch.stack([t['v'].dist().mean for t in out['trace']]).detach().numpy()

        for i in range(mod.I):
            plt.plot(W_trace[:, i, :])
            plt.title('w trace i={}'.format(i+1))
            plt.savefig('{}/W{}_trace.pdf'.format(img_dir, i+1))
            plt.close()

        # Trace for v
        plt.plot(v_trace)
        plt.title('v trace')
        plt.savefig('{}/v_trace.pdf'.format(img_dir))
        plt.close()

        # Plot sig mean trace
        sig2_trace = torch.stack([t['sig2'].dist().mean for t in out['trace']])
        plt.plot(sig2_trace.detach().numpy()[2:])
        plt.title('sig2 trace')
        plt.savefig('{}/sig2_trace.pdf'.format(img_dir))
        plt.close()

        # lam posterior
        def draw_theta():
            return post_proc.sample_params.theta(mod, priors)[0]

        # lam_samps = 100
        print("plot yz", flush=True)
        lam_samps = 30
        lam_draws = [post_proc.lam_post.sample(draw_theta(), mod) for b in range(lam_samps)]
        lam = [torch.stack([lam_b[i] for lam_b in lam_draws]) for i in range(mod.I)]
        lam_est = [lam_i.mode(0)[0] for lam_i in lam]

        # TODO: now, lam in {0, ..., K}
        # See if the other graphs are affected.

        W_mean = W.mean(0)
        Z_mean = Z.mean(0)

        for i in range(mod.I):
            # quiet_yi = y[i][1 - idx_noisy[i], :]
            # quiet_lami = lam_est[i][1 - idx_noisy[i]]
            # plot_yz(quiet_yi, Z_mean, W_mean[i, :], quiet_lami, w_thresh=.05, cm_y=cm, vlim_y=VLIM)
            plt.figure(figsize=(8,8))
            plot_yz(y[i], Z_mean, W_mean[i, :], lam_est[i], w_thresh=.05, cm_y=cm, vlim_y=VLIM)
            # plt.tight_layout()
            # plt.savefig('{}/y{}_post.pdf'.format(img_dir, i + 1), bbox_inches='tight')
            plt.savefig('{}/y{}_post.pdf'.format(img_dir, i + 1), dpi=500)
            plt.close()

        # Plot imputed ys
        for i in range(mod.I):
            yi = vae[i](mod.y_data[i], mod.m[i]).detach()
            # plt.hist(vae[i].mean_fn_cached[mod.m[i]].detach().numpy())
            plt.hist(yi[mod.m[i]].detach().numpy())
            plt.xlim(-10, 5)
            plt.savefig('{}/y{}_imputed_hist.pdf'.format(img_dir, i + 1))
            plt.close()

        plot_dden = True
        if plot_dden:
            # Posterior predictives estimate
            _, y_grid = ddenExpressed.sample(draw_theta())

            # TODO: Consider making this more memory efficient
            #       by just storing (dden_ij_mean, lower, upper)
            print('drawing dden', flush=True)
            dden_draws = 30
            dden_post = [ddenExpressed.sample(draw_theta(), y_grid) for d in range(dden_draws)]

            size_min = 0
            size_max = 300
            size_range = size_max - size_min
            for i in range(mod.I):
                pass
                # TODO: plot dden-expressed
                for j in range(mod.J):
                    dden_ij = torch.stack([dd[i][:, j] for dd in dden_post]).numpy()
                    # dden_ij *= (1 - mod.m[i][:, j].numpy()).mean()
                    dden_ij *= (mod.m[i][:, j].numpy() > 0).mean()
                    dden_ij_mean = dden_ij.mean(0)
                    dden_ij_lower = np.percentile(dden_ij, 2.5, axis=0)
                    dden_ij_upper = np.percentile(dden_ij, 97.5, axis=0)
                    #
                    yij_obs = mod.y_data[i][:, j][1-mod.m[i][:, j]].numpy()
                    # sns.kdeplot(yij_obs[yij_obs > 0], color='lightgrey')
                    sns.kdeplot(yij_obs, color='lightgrey')
                    plt.plot(y_grid.numpy(), dden_ij_mean, color='purple')
                    plt.fill_between(y_grid.numpy(), dden_ij_lower, dden_ij_upper,
                                     alpha=.5, color='purple')
                    plt.title('i: {}, j: {}'.format(i+1, j+1))
                    plt.axvline(0, linewidth=.3, linestyle=':', color='lightgrey')
                    plt.text(.05, .9,
                             'missing: {:.1f}%'.format(mod.m[i][:, j].double().mean() * 100),
                             ha='left', va='center', transform=plt.gca().transAxes)
                    #
                    s1_ij = np.sqrt(eta1_mean[i, j, :]) * size_range + size_min
                    plt.scatter(mu1.mean(0), [0]*mod.L[1], s=s1_ij,
                                alpha=.7, marker='X', linewidth=0, c='red')
                    #
                    plt.savefig('{}/dden-expressed/dden_i{}_j{}.pdf'.format(img_dir, i + 1, j + 1))
                    plt.close()

    print("Done.", flush=True)


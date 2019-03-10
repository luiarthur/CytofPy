import sys
import os

import torch
from torch.distributions.log_normal import LogNormal
from torch.distributions import Gamma

import cytofpy
import lam_post
from plot_yz import add_gridlines_Z, plot_yz
import Timer

import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
import numpy as np
import pickle

# Use smaller learning rate, with double precision
# https://discuss.pytorch.org/t/why-double-precision-training-sometimes-performs-much-better/31194

if __name__ == '__main__':
    torch.set_num_threads(1)

    if len(sys.argv) > 1:
        path_to_exp_results = sys.argv[1]
        SEED = int(sys.argv[2])
    else:
        path_to_exp_results = 'results/sim1-vae/test/'
        SEED = 2

    # subsample = 1.0 # .2
    subsample = .05

    img_dir = path_to_exp_results + '/img/'
    os.makedirs('{}'.format(img_dir), exist_ok=True)
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
    cm = plt.cm.get_cmap('bwr', 9)
    cm.set_bad(color='black')

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
    priors = cytofpy.model.default_priors(y, K=K, L=L,
                                          y_quantiles=[0, 35, 50], p_bounds=[.05, .8, .05])
    priors['sig2'] = LogNormal(-1, .1)
    priors['alpha'] = Gamma(.1, .1)
    priors['delta0'] = Gamma(10, 1)
    priors['delta1'] = Gamma(10, 1)

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

    with Timer.Timer('Model training'):
        out = cytofpy.model.fit(y, max_iter=5000, lr=1e-1, print_freq=10, eps=0,
                                priors=priors, minibatch_size=2000, tau=0.1,
                                trace_every=50, backup_every=50,
                                # verbose=0, seed=SEED, use_stick_break=False)
                                verbose=0, seed=0, use_stick_break=False)
    # Flush output
    sys.stdout.flush()

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))
    show_plots = True
    if show_plots:
        print("Making Plots...")

        # out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

        elbo = out['elbo']
        use_stick_break = out['use_stick_break']
        mod = cytofpy.model.Model(y=y, priors=priors,
                                  tau=out['tau'], use_stick_break=use_stick_break)
        mod.mp = out['mp']
        vae = out['vae']

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
            Z = (v.cumprod(2) > torch.distributions.Normal(0, 1).cdf(H)).numpy()
        else:
            Z = (v > torch.distributions.Normal(0, 1).cdf(H)).numpy()
        plt.imshow(Z.mean(0), aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        add_gridlines_Z(Z[0])
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
        # lam_samps = 100
        lam_samps = 100
        lam = [lam_post.sample(mod) for b in range(lam_samps)]
        lam = [torch.stack([lam_b[i] for lam_b in lam]) for i in range(mod.I)]
        lam_est = [lam_i.mode(0)[0] for lam_i in lam]

        # TODO: TEST
        # I think we can remove noisy cells this way. 
        lam_onehot = []
        idx_noisy = []
        for i in range(mod.I):
            lami_onehot = torch.zeros((lam[i].size(0), lam[i].size(1), K), dtype=torch.int64)
            lam_onehot.append(lami_onehot)
            for b in range(lam_samps):
                lam_onehot[i][b, torch.arange(mod.N[i]), lam[i][b]] = 1
            # Std of lam_i
            idx_noisy.append(lam_onehot[i].double().var(0).sum(1) > .5)
            # quiet cells
            # y[i][1 - idx_noisy[i], :]

        W_mean = W.mean(0)
        Z_mean = Z.mean(0)

        for i in range(mod.I):
            plot_yz(y[i], Z_mean, W_mean[i, :], lam_est[i], w_thresh=.05, cm_y=cm, vlim_y=VLIM)
            plt.tight_layout()
            plt.savefig('{}/y{}_post.pdf'.format(img_dir, i + 1))
            plt.close()

        # Plot imputed ys
        for i in range(mod.I):
            yi = vae[i](mod.y_data[i], mod.m[i]).detach()
            # plt.hist(vae[i].mean_fn_cached[mod.m[i]].detach().numpy())
            plt.hist(yi[mod.m[i]].detach().numpy())
            plt.xlim(-10, 5)
            plt.savefig('{}/y{}_imputed_hist.pdf'.format(img_dir, i + 1))
            plt.close()

    print("Done.")

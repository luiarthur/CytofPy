import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
import blue2red


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


def plot_yz(yi, Z_mean, wi_mean, lami_est, w_thresh=.01,
            cm_greys = plt.cm.get_cmap('Greys', 5),
            cm_y=blue2red.cm(6), vlim_y=(-3, 3), fs_w=10, w_digits=1):
            #cm_y=plt.cm.get_cmap('coolwarm', 7), vlim_y=(-3, 3)):

    J = yi.shape[1]

    vmin_y, vmax_y = vlim_y
    # cm_y.set_bad(color='black')
    # cm_y.set_under(color='blue')
    # cm_y.set_over(color='red')

    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5]) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 2]) 

    # Plot y
    lami_new, counts = relabel_lam(lami_est, wi_mean)
    counts_cumsum = np.cumsum(counts)
    yi_sorted = yi[lami_new.argsort(), :].numpy()

    plt.subplot(gs[0])
    im = plt.imshow(yi_sorted, aspect='auto', vmin=vmin_y, vmax=vmax_y, cmap=cm_y)
    for c in counts_cumsum[:-1]:
        plt.axhline(c, color='yellow')
    plt.xticks(rotation=90)
    plt.xticks(np.arange(J), np.arange(J) + 1)

    ax = plt.gca()
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    cax.xaxis.set_ticks_position("top")
    colorbar(im, cax=cax, orientation="horizontal")

    # Plot Z
    k_ord = wi_mean.argsort()
    z_cols = []

    for k in k_ord.tolist():
        if wi_mean[k] > w_thresh:
            z_cols.append(k)

    z_cols = np.array(z_cols)
    Z_hat = Z_mean[:, z_cols].T

    plt.subplot(gs[1])
    im = plt.imshow(Z_hat, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    ax = plt.gca()
    plt.xticks([])
    plt.yticks(np.arange(len(z_cols)), z_cols + 1, fontsize=fs_w)
    add_gridlines_Z(Z_hat)
    plt.colorbar(orientation='horizontal', pad=.05)

    # add wi_mean on right side
    K = z_cols.shape[0]
    ax2 = ax.twinx()
    ax2.set_yticks(range(K))
    w_perc = wi_mean[z_cols]
    w_perc = [str((wp * 100).round(w_digits)) + '%' for wp in w_perc]
    plt.yticks((K-1) / K * np.arange(K) + .5, w_perc[::-1], fontsize=fs_w)
    plt.yticks()
    ax2.tick_params(length=0)

    fig = plt.gcf()
    fig.subplots_adjust(hspace=0.2)

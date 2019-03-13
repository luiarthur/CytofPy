import matplotlib.pyplot as plt
from matplotlib import gridspec
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
            cm_y=blue2red.cm(6), vlim_y=(-3, 3)):
            #cm_y=plt.cm.get_cmap('coolwarm', 7), vlim_y=(-3, 3)):

    J = yi.shape[1]

    vmin_y, vmax_y = vlim_y

    # cm_y.set_bad(color='black')
    # cm_y.set_under(color='blue')
    # cm_y.set_over(color='red')

    k_ord = wi_mean.argsort()
    z_cols = []

    for k in k_ord.tolist():
        if wi_mean[k] > w_thresh:
            z_cols.append(k)

    z_cols = np.array(z_cols)
    Z_hat = Z_mean[:, z_cols]
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5]) 
    plt.subplot(gs[0])

    plt.imshow(Z_hat, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    plt.colorbar()
    plt.xticks(np.arange(len(z_cols)), z_cols + 1)
    plt.yticks(np.arange(J), np.arange(J) + 1)
    add_gridlines_Z(Z_hat)

    lami_new, counts = relabel_lam(lami_est, wi_mean)
    counts_cumsum = np.cumsum(counts)

    plt.subplot(gs[1])
    yi_sorted = yi[lami_new.argsort(), :].numpy().T
    plt.imshow(yi_sorted, aspect='auto', vmin=vmin_y, vmax=vmax_y, cmap=cm_y)
    for c in counts_cumsum[:-1]:
        plt.axvline(c, color='yellow')

    plt.colorbar()
    plt.yticks([])

import os
import sys
import cytofpy
import Timer
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib 
matplotlib.use('Agg')

import numpy as np
if __name__ == '__main__':
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

    np.random.seed(0)

    # Read Data
    data = cytofpy.util.readCB(path_to_cb_data)
    I = len(data['y'])

    y = data['y']
    Y_all = np.concatenate(y, 0)

    # remove markers that are highly missing/negative or positive
    good_markers = [True, False, True, False, True, False, True, True, False,
                    True, False, True, True, True, False, True, True, False, False,
                    True, False, True, True, True, True, False, True, False, True,
                    True, False, True]
    is_good_marker = np.where(good_markers)

    Y = Y_all[:, is_good_marker].squeeze()

    idx_nan = np.isnan(Y)
    Y[idx_nan] = np.random.randn(*Y.shape)[idx_nan] - 3

    n_subsample = 2000
    idx_subsample = np.random.choice(Y.shape[0], n_subsample, replace=False)
    # idx_subsample = np.random.choice(y[0].shape[0], n_subsample, replace=False)

    tsne = TSNE()
    tsne.fit(Y[idx_subsample])

    plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], s=10)
    plt.show()

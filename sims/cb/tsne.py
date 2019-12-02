import os
# Limit the number of Openblas threads used
# export OMP_NUM_THREADS=4
# Needs to be done before `import numpy`
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import cytofpy
import Timer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
mpl.use('Agg')

# Interactive plot
# mpl.use("TkAgg")

# Non-interactive plot 
# mpl.use("Agg")

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
    Y_scaled = scale(Y)

    # pca = PCA(n_components=2)
    # pca.fit(Y.T)
    # plt.scatter(pca.components_[2, :], pca.components_[1, :])
    # plt.show()

    n_subsample = 500
    idx_subsample = np.random.choice(Y_scaled.shape[0], n_subsample, replace=False)
    # idx_subsample = np.random.choice(y[0].shape[0], n_subsample, replace=False)

    # idx_subsample = np.arange(Y_scaled.shape[0])  # All CB samples

    # tsne = TSNE(verbose=1, method='exact', random_state=0)
    tsne = TSNE(verbose=1, n_iter=3000)
    with Timer.Timer(ndigits=2):
        tsne.fit(Y_scaled[idx_subsample])

    tsne.kl_divergence_

    plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], s=10)
    plt.show()
    plt.close()

    # # Processing times for exact method (seconds)
    # proc_time_exact = {100: 1.44,
    #                    200: 3.17,
    #                    300: 8.24,
    #                    400: 12.53,
    #                    500: 16.32,
    #                    1000: 45.41,
    #                    5000: 902.99} # 4 cores,
    #                    # 10000: None,
    #                    # 50000: None}
    #                    # 5000: 1086.99} # 32 cores,

    # plt.plot(list(proc_time_exact.keys()),
    #          list(proc_time_exact.values()),
    #          marker='o'); plt.show()

    # # Processing times (seconds)
    # proc_time = {100: 1,
    #              500: 3,
    #              1000: 7,
    #              1500: 13,
    #              2000: 16,
    #              2500: 18,
    #              3000: 25,
    #              5000: 46,
    #              10000: 92,
    #              20000: 213,
    #              50000: 708,
    #              57105: 820}

    # plt.plot(list(proc_time.keys()),
    #          list(proc_time.values()),
    #          marker='o'); plt.show()

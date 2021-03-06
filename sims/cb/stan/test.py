import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan
from pystan_vb_extract import pystan_vb_extract

import impute

# Compile STAN model
sm = pystan.StanModel(file='model_no_missing.stan')

# Data
data = pd.read_csv('../data/cb.csv')
Y = data.drop(columns=['sample_id']).to_numpy()
for i in data.sample_id.drop_duplicates():
    idx = np.where(data.sample_id == i)[0]
    Y[idx], _ = impute.impute_all(Y[idx], np.isnan(Y[idx]),
                                  max_iter=10, tol=1e-3, seed=0)

# Use subsample of data,
# because STAN can't handle all the data ...
np.random.seed(0)
num_samps = 500
idx = np.random.choice(Y.shape[0], num_samps, replace=False)

model_data = dict(
    J = Y.shape[1],
    I = len(data.sample_id.drop_duplicates()),
    K = 20,
    L0 = 5,
    L1 = 3,
    N = Y[idx].shape[0],
    group = (data.sample_id + 1).to_numpy()[idx],
    y = Y[idx]
)

# Fit STAN model (VB)
fit_vb = sm.vb(data=model_data, iter=1000, seed=1)
vb_results = pystan_vb_extract(fit_vb)

# Plots
make_plots = False
if make_plots:
    plt.boxplot(vb_results['W'][:, 0, :], showfliers=False, showmeans=True)
    plt.show()

    plt.boxplot(vb_results['sigma'], showfliers=False, showmeans=True)
    plt.show()

    plt.boxplot(vb_results['mu0'], showfliers=False, showmeans=True)
    plt.show()

    plt.boxplot(vb_results['mu1'], showfliers=False, showmeans=True)
    plt.show()

    plt.hist(vb_results['alpha'], showfliers=False, showmeans=True)
    plt.show()

# Fit STAN model (NUTS)
# The link below describes the tuning parameters:
# https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
# fit = sm.sampling(data=model_data, iter=200, chains=1, seed=1, warmup=100,
#                   control=dict(max_treedepth=5))

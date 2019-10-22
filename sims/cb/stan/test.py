import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

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

# Use subsample of data, because STAN can't
# handle all the data ...
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

# Fit STAN model
fit = sm.sampling(data=model_data, iter=200, chains=1, seed=1)
                  # control=dict(max_treedepth=5), algorithm='HMC')

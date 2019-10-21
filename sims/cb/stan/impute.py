import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import trange


def impute(Y, M, k):
    """
    Impute values in matrix Y.
    M is a missingness indicator matrix.
    (M[i, j] == 1) => Y[i, j] is missing.
    (M[i, j] == 0) => Y[i, j] is observed.
    k is the column to impute missing values for.
    """
    # get indices for observed y in column k
    idx_obs = np.where((M[:, k] == False) * (Y[:, k] < 0))[0]

    # get indices for missing y in column k
    idx_mis = np.where(M[:, k] == True)[0]

    # Fit model
    model = LinearRegression()
    response = Y[idx_obs, k].reshape(-1, 1)
    model.fit(np.delete(Y[idx_obs, :], k, axis=1), response)

    # make predictions
    pred = model.predict(np.delete(Y[idx_mis, :], k, axis=1))

    # Fill matrix
    Y[idx_mis, k] = pred.squeeze()


def impute_all(Y, M, max_iter=30, tol=1e-3):
    # Make a copy
    X = Y + 0

    # number of columns
    K = X.shape[1]

    # initialize
    for k in range(K):
        num_missing = M[:, k].sum()
        neg_obs_idx = np.where(X[:, k] < 0)[0]
        X[M[:, k], k] = np.random.choice(X[neg_obs_idx, k], size=num_missing,
                                         replace=True)

    # Initialize the change in the matrices
    diff_x = []

    for i in trange(max_iter):
        X_old = X + 0
        for k in range(K):
            impute(X, M, k)
        diff = np.abs(X_old - X).mean()
        diff_x.append(diff)
        if diff < tol:
            print('Convergence detected! Stopping early.')
            break
    
    return X, diff_x



if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt 
    np.random.seed(0)

    dat = pd.read_csv('../data/cb.csv')
    Y1 = dat[dat.sample_id == 1].drop(columns=['sample_id']).to_numpy()
    M1 = np.isnan(Y1)

    Y1_new, diffs = impute_all(Y1, M1, max_iter=10, tol=1e-4)

    # Visualize
    # j = 7; plt.hist(Y1_new[np.where(M1[:, j])[0], j], bins=50); plt.show()
    # plt.hist(Y1[np.where(1-M1[:, j])[0], j], bins=50); plt.show()
    # plt.hist(Y1_new[:, j], bins=50); plt.show()

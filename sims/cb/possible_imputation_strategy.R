# Possible imputation strategy

Y = as.matrix(read.table('data/cb.txt', skip=2))
M = is.na(Y)


y1 = Y[1:41474, ]
m1 = is.na(y1)

# TODO: use a neural net

# Impute missing values in a column
impute = function(Yi, Mi, k) {
  # get indices for observed Y in column k
  idx_obs = which(Mi[, k] == 0)

  # get indices for missing Y in column k
  idx_miss = which(Mi[, k] == 1)

  # Fit model to predict observed Y in column k
  # given Y in other columns (univariate multiple linear regression)
  model = lm(Yi[idx_obs, k] ~ Yi[idx_obs, -k])

  # Get coefficients from trained model
  beta = model$coef[-1]

  # Get intercept  from trained model
  intercept = model$coef[1]

  # Make predictions for missing Y in column k
  pred = Yi[idx_miss, -k] %*% beta + intercept

  return(pred)
}


# Impute all missing values in a matrix
impute_all = function(Yi, Mi, max_iter=30, tol=1e-3) {
  X = y1
  # X[m1] = rnorm(sum(m1), -3, sd=.1)
  K = NCOL(X)
  # Initialize
  for (k in 1:K) {
    num_missing = sum(m1[, k])
    neg_obs_idx = which(X[, k] < 0)
    X[m1[, k], k] = sample(X[neg_obs_idx, k], replace=TRUE, size=num_missing)
  }

  diff_X = c()
 
  for (i in 1:max_iter) {
    X_old = X
    cat('\r', i, '/', max_iter)
    for (k in 1:K) {
      X[m1[, k], k] = impute(X, m1, k)
    }
    diff_X = c(diff_X, mean(abs(X_old[m1] - X[m1])))
    if (tail(diff_X, 1) < tol) {
      print('Convergence detected! Stopping early.')
      break
    }
  }

  return(list(X=X, diff_X=diff_X))
}


out = impute_all(y1, m1, max_iter=10)
y1_new = out$X
plot(out$diff_X, type='o')
hist(y1_new[,4])

is_good_row = function(x) all(abs(x) < 6)
y1_new_good_rows = apply(y1_new, 1, is_good_row)
y1_new_good = y1_new[y1_new_good_rows, ]
m1_good = m1[y1_new_good_rows, ]

plotme = function(j) {
  par(mfrow=c(1, 3))
  hist(y1_new_good[m1_good[, j], j], main="new missing only")
  hist(y1_new_good[, j], main="new")
  hist(y1[, j], main="old")
  par(mfrow=c(1, 1))
}

plotme(9)

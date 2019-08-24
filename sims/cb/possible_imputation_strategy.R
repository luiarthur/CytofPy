# Possible imputation strategy

Y = as.matrix(read.table('data/cb.txt', skip=2))
M = is.na(Y)


y1 = Y[1:41474, ]
m1 = is.na(y1)


impute = function(Yi, Mi, k) {
  # idx_obs = which(Mi[, k] == 0 & Yi[, k] < 0)
  idx_obs = which(Mi[, k] == 0)
  idx_miss = which(Mi[, k] == 1)
  model = lm(Yi[idx_obs, k] ~ Yi[idx_obs, -k])
  beta = model$coef[-1]
  intercept = model$coef[1]
  pred = Yi[idx_miss, -k] %*% beta + intercept
  return(pred)
}


impute_all = function(Yi, Mi, max_iter=30, tol=1e-3) {
  X = y1
  X[m1] = rnorm(sum(m1), -3, sd=.1)
  K = NCOL(X)
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

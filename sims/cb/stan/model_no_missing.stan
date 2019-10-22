data {
  int J;
  int I;
  int K;
  int L0;
  int L1;
  int N;
  int<lower=1> group[N]; 
  real y[N, J];
}

transformed data {
  vector<lower=0>[K] d_W = rep_vector(1.0 / K, K);
  vector<lower=0>[L0] d_eta0 = rep_vector(1.0 / L0, L0);
  vector<lower=0>[L1] d_eta1 = rep_vector(1.0 / L1, L1);
}

parameters {
  simplex[K] W[I];
  vector<lower=0>[L0] delta0;
  vector<lower=0>[L1] delta1;
  simplex[L0] eta0[I, J];
  simplex[L1] eta1[I, J];
  vector<lower=0, upper=1>[K] v;
  vector<lower=0>[I] sigma;
  real<lower=0> alpha;
}

transformed parameters {
  vector<upper=0>[L0] mu0 = -cumulative_sum(delta0);
  vector<lower=0>[L1] mu1 = cumulative_sum(delta1);
}

model {
  vector[N] ll;

  // Priors
  sigma ~ gamma(3, 2);
  alpha ~ gamma(1, 1);
  delta0 ~ gamma(3, 2);
  delta1 ~ gamma(3, 2);
  v ~ beta(alpha / K, 1);

  for (i in 1:I) {
    W[i] ~ dirichlet(d_W);
    for (j in 1:J) {
      eta0[i, j] ~ dirichlet(d_eta0);
      eta1[i, j] ~ dirichlet(d_eta1);
    }
  }

  // Model
  for (n in 1:N) {
    real lmix1;
    real lmix0;
    vector[L0] kernel0;
    vector[L1] kernel1;
    int i = group[n];
    vector[K] ll_n = rep_vector(0.0, K);

    for (k in 1:K) {
      for (j in 1:J) {
        kernel0 = log(eta0[i, j]) + normal_lpdf(y[n, j] | mu0, sigma[i]);
        kernel1 = log(eta1[i, j]) + normal_lpdf(y[n, j] | mu1, sigma[i]);

        lmix0 = log_sum_exp(kernel0);
        lmix1 = log_sum_exp(kernel1);

        ll_n[k] += log_mix(v[k], lmix1, lmix0);
      }
    }

    // target += log_sum_exp(log(W[i]) + ll_n);
    ll[n] = log_sum_exp(log(W[i]) + ll_n);
  }

  target += sum(ll);
}

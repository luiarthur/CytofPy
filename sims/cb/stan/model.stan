data {
  int J;
  int I;
  int K;
  int L;
  int N;
  int<lower=1> group[N]; 
  real y[N, J];
  int<lower=0, upper=1> m[N, J];
}

parameters {
  simplex[K] W[I];
  vector<lower=0>[L] delta0;
  vector<lower=0>[L] delta1;
  simplex[L] eta0[I, J];
  simplex[L] eta1[I, J];
  vector<lower=0, upper=1>[K] v;
  vector<lower=0>[I] sigma;
  real<lower=0> alpha;
}

transformed parameters {
  vector<upper=0>[L] mu0;
  vector<lower=0>[L] mu1;

  mu0 = -cumulative_sum(mu0)[1:L];
  mu1 = cumulative_sum(mu1)[1:L];
}

model {
  // Priors
  sigma ~ gamma(3, 2);
  alpha ~ gamma(3, 2);
  delta0 ~ gamma(3,1);
  delta1 ~ gamma(3,1);
  v ~ beta(alpha / K, 1);

  for (i in 1:I) {
    W[i] ~ dirichlet(1 / K);
    for (j in 1:J) {
      eta0 ~ dirichlet(1 / L);
      eta1 ~ dirichlet(1 / L);
    }
  }

}

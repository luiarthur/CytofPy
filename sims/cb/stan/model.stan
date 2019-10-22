/* NOTE:
    - A `generated quantities` block can be used to compute deviance / LPML.
      They are computed only after a sample has been generated. 
      They are printed as output.
      See: 
        https://mc-stan.org/docs/2_18/reference-manual/program-block-generated-quantities.html
    - See this for transformed parameters: 
        https://mc-stan.org/docs/2_18/stan-users-guide/change-point-section.html
    - See this for missing values:
        https://mc-stan.org/docs/2_20/stan-users-guide/missing-multivariate-data.html
 */

functions {
  /* See this:
     https://mc-stan.org/docs/2_20/stan-users-guide/basic-functions-section.html
   */
  int num_y_obs(real m[,]) {
    return sum(m);
  }

  int num_y_mis(real m[,]) {
    return num_elements(m) - num_y_obs(m);
  }
}

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

transformed data {
  real y_obs[num_y_obs(m)]; 
  // TODO: define y_obs

  /* ALSO need an index transformer.
     1. [n, j] -> idx in y_obs
     2. [n, j] -> idx in y_mis
   */
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
  vector<upper=0>[L] mu0 = -cumulative_sum(mu0)[1:L];
  vector<lower=0>[L] mu1 = cumulative_sum(mu1)[1:L];

  real y_mis[num_y_mis(m)]; 
  // TODO: define y_mis
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

  // Model
  for (n in 1:N) {
    for (j in 1:J) {
      // TODO
      if (M[n, j]) {
        /* Missing data. Use observed likelihood as prior and 
           Bernoulli missing mechanism for likelihood.
         */
      } else {
        /* Observed data. Use observed likelihood only. */
      }
    }
  }
}

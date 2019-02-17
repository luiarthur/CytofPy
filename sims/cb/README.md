# Remarks

When the imputed values are encouraged to be small and far from bulk of the
data, Z tends to have more features. When they are encouraged to be in the
IQR or the observed data, Z tends to have fewer non-zero columns.
The model is not very sensitive to the prior for alpha.

Constraining sig0 and sig1 to be between 0 and 1 has some benefits. But, the
model seems to prefer larger sigma. Perhaps more L components are needed.

But using many L-components tends to make Z have fewer non-zero columns, 
while using fewer L-components leads to more Z columns.

For MCMC, it seems imputing values nearer to bulk of data helps better
the fit of the observed data.

**Maybe** I should use double instead of float?
I tried changing everything to double. That made the runs the same on
local machine and on server. But, elbo's became nan in 70 iterations
no matter what I did. I changed it back to float.

Perhaps this has something to do with gradients of `rsamples`. Perhaps
with double and my advi models, the stability is best. IDK.

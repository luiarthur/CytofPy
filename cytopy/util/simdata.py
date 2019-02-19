import numpy as np
import torch

from torch.distributions import Dirichlet
from torch.distributions import Categorical
from torch.distributions import Bernoulli
from torch.distributions import Beta
from torch.distributions import Normal

# CHECK THIS!
def simdata(N=[300, 100, 200], J=25, a_W=[200., 500., 200., 100.], L0=5, L1=3, sig=None, alpha=None):
    I = len(N)
    data = {'y': [], 'm': []}
    K = len(a_W)
    a_W = torch.tensor(a_W).double()

    if alpha is None:
        alpha = K

    v = Beta(alpha / K, 1).sample((K, ))
    if J % K == 0 and J > K:
        g = int(J / K)
        Z = torch.tensor(np.kron(torch.eye(K), torch.ones(g)).T)
    else:
        Z = Bernoulli(v).sample((J, )).reshape(J, K)

    a_eta0 = torch.ones(L0) / L0
    a_eta1 = torch.ones(L1) / L1

    W = Dirichlet(a_W).sample((I, ))
    for i in range(I):
        W[i, :] = W[i, torch.randperm(K)]

    eta0 = Dirichlet(a_eta0).sample((I, J))
    eta1 = Dirichlet(a_eta1).sample((I, J))

    if sig is None:
        sig = torch.ones((I, )) * 1.0

    mu0 = -(torch.arange(L0) + 3 * sig.max()).double()
    mu1 = (torch.arange(L1) + 3 * sig.max()).double()

    params = {'W': W, 'v': v, 'eta0': eta0, 'eta1': eta1,
              'mu0': mu0, 'mu1': mu1, 'sig': sig, 'Z': Z}

    lam = []
    for i in range(I):
        # wi = Dirichlet(a_W).sample()
        wi = W[i, :].squeeze()

        gam0i = []
        gam1i = []
        for j in range(J):
            gam0i.append(Categorical(eta0[i, j, :]).sample((N[i], )))
            gam1i.append(Categorical(eta1[i, j, :]).sample((N[i], )))

        gam0i = torch.stack(gam0i, 1)
        gam1i = torch.stack(gam1i, 1)

        lami = Categorical(wi).sample((N[i], )).sort()[0]
        lam.append(lami)

        Zi = Z[:, lami]
        Zi.transpose_(0, 1)

        mui = (1 - Zi) * mu0[gam0i] + Zi * mu1[gam1i]

        yi = Normal(mui, sig[i]).sample()

        data['y'].append(yi + 0)
        data['m'].append(yi + 0)

    params['lam'] = lam

    return {'data': data, 'params': params}

# data = simdata()

def simdata_with_missing_values(N=[300, 100, 200], J=25,
                                a_W=[200., 500., 200., 100.],
                                L0=5, L1=3, sig=None, alpha=None):
    data = simdata(N=N, J=J, a_W=a_W, L0=L0, L1=L1, sig=sig, alpha=alpha)

    I = len(data['y'])
    for i in range(I):
        pass

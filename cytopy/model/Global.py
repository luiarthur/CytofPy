import torch
from torch.distributions import Normal, Gamma, Dirichlet, Beta
from torch.distributions.kl import kl_divergence as kld
from cytopy.vardist import VDGamma, VDNormal, VDBeta, VDDirichlet, VI

def compute_Z(logit, tau):
    """
    This enables the backward gradient computations via Z.
    Notice at the end, we basically return Z (binary tensor).
    But we make use of the smoothed Z, which is differentiable.
    We detach (Z - smoothed_Z) so that the gradients are not 
    computed, and then add back smoothed_Z for the return.
    The only gradient will then be that of smoothed Z.
    """
    smoothed_Z = (logit / tau).sigmoid()
    Z = (smoothed_Z > 0.5).float()
    return (Z - smoothed_Z).detach() + smoothed_Z


def default_priors(y, K:int=30, L=None):
    I = len(y)

    J = y[0].size(1)
    for i in range(I):
        assert(y[i].size(1) == J)

    N = [y[i].size(0) for i in range(I)]

    if L is None:
        L = [5, 5]

    K = K

    return {'I': I, 'J': J, 'N': N, 'L': L, 'K': K,
            'mu0': Gamma(torch.ones(L[0]), torch.ones(L[0])),
            'mu1': Gamma(torch.ones(L[1]), torch.ones(L[1])),
            #
            'sig0': Gamma(torch.ones(L[0]), torch.ones(L[0])),
            'sig1': Gamma(torch.ones(L[1]), torch.ones(L[1])),
            #
            'eta0': Dirichlet(torch.ones(L[0]) / L[0]),
            'eta1': Dirichlet(torch.ones(L[1]) / L[1]),
            #
            'alpha': Gamma(.1, .1),
            'H': Normal(0, 1),
            #
            'beta0': Gamma(1, 1),
            'beta1': Gamma(1, 1),
            #
            'W': Dirichlet(torch.ones(K) / K)}

class Global(VI):
    def __init__(self, priors, iota=0.5, verbose=1):
        self.verbose = verbose

        # Dimensions of data
        self.I = priors['I']
        self.J = priors['J']
        self.N = priors['N']
        self.Nsum = sum(self.N)

        # Tuning Parameters
        self.iota = iota

        # Dimensions of parameters
        self.L = priors['L']
        self.K = priors['K']

        # Assign variational distributions
        self.mu0 = VDGamma((self.L[0]))
        self.mu1 = VDGamma((self.L[1]))
        self.sig0 = VDGamma((self.L[0]))
        self.sig1 = VDGamma((self.L[1]))
        self.eta0 = VDDirichlet((self.I, self.J, self.L[0]))
        self.eta1 = VDDirichlet((self.I, self.J, self.L[1]))
        self.W = VDDirichlet((self.I, self.K))
        self.alpha = VDGamma(1)
        self.beta0 = VDGamma((self.I, self.J))
        self.beta1 = VDGamma((self.I, self.J))
        self.v = VDBeta(self.K)
        self.H = VDNormal((self.J, self.K))

        # This must be done after assigning variational distributions
        super(Global, self).__init__()

    def loglike(self, y, params):
        for i in range(self.I):
            # Y: Ni x J
            # muz: Lz
            # etaz_i: 1 x J x Lz

            # Ni x J x Lz
            d0 = Normal(-self.iota - params['mu0'].cumsum(0)[None, None, :],
                        params['sig'][i]).log_prob(y[i][:, :, None])
            d0 += params['eta0'][i:i+1, :, :].log()

            d1 = Normal(self.iota + params['mu1'].cumsum(0)[None, None, :],
                        params['sig'][i]).log_prob(y[i][:, :, None])
            d1 += params['eta1'][i:i+1, :, :].log()
            
            # Ni x J
            logmix_L0 = torch.logsumexp(d0, 2)
            logmix_L1 = torch.logsumexp(d1, 2)

            # Z: J x K
            # H: J x K
            # v: K
            # c: Ni x J x K
            # d: Ni x K
            # Ni x J x K

            b_vec = params['v'].cumprod(0)
            Z = compute_Z(b_vec[None, :] - Normal(0, 1).cdf(params['H']), self.tau)
            c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
            d = c.sum(1)

            f = d + params['W'][i:i+1, :].log()
            lli = torch.logsumexp(f, 1).mean(0) * (self.N[i] / self.Nsum)
            assert(lli.dim() == 0)

            ll += lli

        if self.verbose >= 2:
            print('log_like: {}'.format(ll))

        return ll

    def kl_qp(self, params):
        res = 0.0

        for key in self.param:
            res += kld(self.params[key].dist(), self.priors[key]).sum()

        res += kld(self.v.dist(), Beta(params['alpha'], 1)).sum()

        return res / self.Nsum

    def forward(self, y):
        params = self.sample_params()
        elbo = self.loglike(y, params) - self.kl_qp(params)
        return elbo



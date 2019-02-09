import torch
from .Global import Global as GlobalMod
from .Local import Local as LocalMod

def fit(y, minibatch_size, priors=None, max_iter=1000, lr=1e-1, print_freq=10):
    I = len(y)
    m = [None]

    gmodel = GlobalMod(priors)
    lmodel = LocalMod()

    elbo_hist = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    for t in range(max_iter):
        # elbo = gmodel(y, X)
        # loss = -elbo / N
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        elbo_hist.append(-loss.item())
        if t % print_freq == 0:
            print('iteration: {} / {} | elbo: {}'.format(t, max_iter, elbo_hist[-1]))

        if t > 10 and abs(elbo_hist[-1] / elbo_hist[-2] - 1) < 1e-4:
            print('Convergence suspected! Ending optimizer early.')
            break




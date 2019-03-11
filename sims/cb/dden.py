import torch
import loglike

import importlib
importlib.reload(loglike)

def sample(mod, y=None):
    if y is None:
        upper = 6
        lower = -6
        grid_size = 100
        step = (upper - lower) / grid_size
        yi = torch.arange(start=-6, end=6, step=step)[:, None] * torch.ones((1, mod.J))
        y = [yi for i in range(mod.I)]

    # TODO: change loglike.py and lam_post.py to index the output in loglike.py by 
    #       (i, n, j, k); instead of (i, n, k). i.e. the output should be 
    #       (I, Ni, J, K); instead of (I, Ni, K). Think if this makes sense.
    
    dden = loglike.sample(mod, y, each_marker=True)
    return (dden, y)

# TEST
# dden, y = sample(mod, y=mod.y_data)
# dden, y = sample(mod)
# i = 0; j = 0
# obs_ij = 1 - mod.m[i][:, j]
# plt.plot(y[i][:, j].numpy(), dden[i][:, j].exp().numpy())
# plt.hist(mod.y_data[i][obs_ij, j].numpy(), density=True)
# plt.xlim(-10, 4)
# plt.show()

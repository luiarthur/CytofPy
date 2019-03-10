import torch
import loglike

def sample(mod, y=None):
    if y is None:
        upper = 6
        lower = -6
        grid_size = 100
        step = (upper - lower) / grid_size
        yi = torch.arange(start=-6, end=6, step=step)[:, None] * torch.ones((1, mod.J))
        y = [yi for i in range(mod.I)]

    dden = []
    # TODO: change loglike.py and lam_post.py to index the output in loglike.py by 
    #       (i, n, j, k); instead of (i, n, k). i.e. the output should be 
    #       (I, Ni, J, K); instead of (I, Ni, K). Think if this makes sense.

    

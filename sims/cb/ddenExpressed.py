import torch
from torch.distributions import Normal

def sample(theta, y_grid=None):
    return_y = False

    if y_grid is None:
        upper = 6
        lower = -6
        grid_size = 100
        step = (upper - lower) / grid_size
        y_grid = torch.arange(start=-6, end=6, step=step)
        return_y = True

     mu1 = theta['mu1']
     eta1 = theta['eta1']
     sig = theta['sig']
     I = len(theta['y'])

     out = []
     for i in range(I):
         tmp = eta1[i:i+1, :, :].log()
         tmp += Normal(mu1[None, None, :], sig[i]).log_prob(y_grid[None, None, :])
         out.append(tmp)

     if return_y:
       return out, y_grid
     else:
       return out

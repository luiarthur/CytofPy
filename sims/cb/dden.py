import torch
import loglike
import gam_post
import util
from torch.distributions import Normal

def sample(mod, lam_draw, y_grid=None):
    if y_grid is None:
        upper = 6
        lower = -6
        grid_size = 100
        step = (upper - lower) / grid_size
        y_grid = torch.arange(start=-6, end=6, step=step)

    # TODO: TEST
    gam, mu, sig = gam_post.sample(mod, lam_draw)

    dden = []
    for i in range(mod.I):
        gami_onehot = util.get_one_hot(gam[i], sum(mod.L))
        obs_i = 1 - mod.m[i]
        mu_i = (gami_onehot * mu[None, None, :]).sum(-1)
        dden_i = Normal(mu_i[:, :, None], sig[i]).log_prob(y_grid[None, None, :]).exp()
        dden_i = dden_i * obs_i[:, :, None].double()
        dden_i = dden_i.sum(0) / obs_i.sum(0, keepdim=True).double().transpose(0, 1)
        dden.append(dden_i)

    return (y_grid, dden)

# TODO: TEST
# import seaborn as sns
# lam_draw = [lami[0] for lami in lam]
# y_grid, dden = dden.sample(mod, lam_draw)
# for i in range(mod.I):
#     for j in range(mod.J):
#         plt.plot(y_grid.numpy(), dden[i][j].numpy())
#         sns.kdeplot(mod.y_data[i][:, j][1-mod.m[i][:, j]].numpy(), color='lightgrey')
#         plt.title('i: {}, j: {}'.format(i+1, j+1))
#         plt.axvline(0, linestyle='--', color='lightgrey')
#         plt.show()
# 
# 

import torch
from torch.distributions import Normal

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

class VAE(torch.nn.Module):
    def __init__(self, J, mean_init=-3, sd_init=.1):
        super(VAE, self).__init__()

        # Variational parameters
        self.mean = torch.nn.Parameter(torch.ones((1, J)) * mean_init)
        self.log_sd = torch.nn.Parameter((torch.ones((1, J)) * sd_init).log())
        # self.mean = torch.nn.Parameter(torch.ones((1,  )) * mean_init)
        # self.log_sd = torch.nn.Parameter((torch.ones((1,  )) * sd_init).log())

    def forward(self, y_mini, m_mini, N_full):
        # Set missing y to 0 (to ensure not nan)
        y_mini = y_mini + 0
        y_mini[m_mini] = 0

        # Set missing indicators to double
        mi_double = m_mini.double()

        # Mean function
        mean_fn = y_mini * (1 - mi_double) + self.mean * mi_double

        # SD function
        # sd_fn = self.log_sd.exp() * mi_double
        sd_fn = (self.log_sd.sigmoid()*.3) * mi_double

        # imputed y
        y_imputed = Normal(mean_fn, sd_fn).rsample()

        # log_qy
        log_qy = Normal(mean_fn[m_mini], sd_fn[m_mini]).log_prob(y_imputed[m_mini]).sum()

        # batch size
        batchsize = y_mini.size(0)

        return y_imputed, log_qy * (N_full / batchsize)

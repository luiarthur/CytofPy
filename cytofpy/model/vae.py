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

        # Cached mean and sd. Be cautious when using these!
        self.mean_fn_cached = None
        self.sd_fn_cached = None

    def dist(self, y_mini, m_mini):
        # Set missing y to 0 (to ensure not nan)
        y_mini = y_mini + 0
        y_mini[m_mini] = 0

        # Set missing indicators to double
        m_mini = m_mini.double()

        # Mean function
        mean_fn = y_mini * (1 - m_mini) + self.mean * m_mini

        # SD function
        sd_fn = self.log_sd.exp() * m_mini

        # Cache mean and sd fn. Be cautious when using these!
        self.mean_fn_cached = mean_fn + 0
        self.sd_fn_cached = sd_fn + 0

        return Normal(mean_fn, sd_fn)

    def forward(self, y_mini, m_mini):
        y_imputed = self.dist(y_mini, m_mini).rsample()
        return y_imputed

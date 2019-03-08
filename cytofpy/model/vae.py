import torch
from torch.distributions import Normal
from torch.nn import Parameter, ParameterList

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

class VAE(torch.nn.Module):
    def __init__(self, J, hidden_size=None, y_min=-4, y_max=-2, s_min=.1, s_max=.3):
        super(VAE, self).__init__()

        input_size = J
        output_size = J # J-dimensional y_missing for some (i, n)

        if hidden_size is None:
            # hidden_size = 2 * input_size
            hidden_size = input_size * 2

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2_m = torch.nn.Linear(hidden_size, output_size)

        self.fc2_s = torch.nn.Linear(hidden_size, 1)

        self.act_fn = torch.nn.Tanh()
        # self.act_fn = torch.nn.ReLU()
        # self.act_fn = torch.nn.Sigmoid()

        self.m = None
        self.s = None

        self.y_min = y_min
        self.y_max = y_max
        self.y_range = self.y_max - self.y_min

        self.s_min = s_min
        self.s_max = s_max
        self.s_range = self.s_max - self.s_min

    def forward(self, y_in, m_in):
        N = y_in.size(0)

        # make a copy of y_in
        y_in = y_in + 0.
        y_in[m_in] = 0
        yi = y_in + 0.
        # set missing values to 0
        yi[m_in] = 0.
        # make a copy of m_in
        mi = m_in.double()

        # x = self.fc1(torch.cat([y_in, m_in], dim=1))
        # x = self.fc1(torch.cat([y_in, i_idx * torch.ones(N, self.I)], dim=1))
        x = self.fc1(yi)

        # FIXME: remove these clamps?
        m = self.act_fn(x)
        m = self.fc2_m(m).sigmoid() * self.y_range + self.y_min

        log_s = self.act_fn(x)
        log_s = self.fc2_s(log_s)
        s = log_s.sigmoid() * self.s_range + self.s_min

        self.m = m
        self.s = s

        yi = Normal(self.m, self.s).rsample()
        
        # NOTE: A trick to prevent computation of gradients for
        #       imputed observed values
        return yi - (1 - mi) * yi.detach() + (1 - mi) * y_in


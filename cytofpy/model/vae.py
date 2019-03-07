import torch
from torch.distributions import Normal
from torch.nn import Parameter, ParameterList

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

class VAE(torch.nn.Module):
    def __init__(self, J, hidden_size=None):
        super(VAE, self).__init__()

        # input_size = 2 * J
        input_size = J
        output_size = J # J-dimensional y_missing for some (i, n)

        if hidden_size is None:
            # hidden_size = 2 * input_size
            hidden_size = 10

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2_m = torch.nn.Linear(hidden_size, output_size)
        self.fc2_s = torch.nn.Linear(hidden_size, output_size)

        # self.act_fn = torch.nn.Tanh()
        self.act_fn = torch.nn.ReLU()
        # self.act_fn = torch.nn.Sigmoid()

        self.m = None
        self.s = None

    def forward(self, y_in, m_in):
        N = y_in.size(0)

        # x = self.fc1(torch.cat([y_in, m_in], dim=1))
        x = self.fc1(y_in)

        m = self.act_fn(x)
        m = self.fc2_m(m)

        log_s = self.act_fn(x)
        log_s = self.fc2_s(log_s)
        s = log_s.exp()

        self.m = m
        self.s = s

        # FIXME: remove the clamps
        # self.m = (m * m_in).clamp(-6, 0) + y_in * (1 - m_in)
        # self.s = (s * m_in).clamp(0, 2) + (-100.0) * (1 - m_in)
        self.m = self.m.clamp(-4, -2)
        self.s = self.s.clamp(.5, 2)

        return Normal(self.m, self.s).rsample()

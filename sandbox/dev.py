import torch
from torch.nn import Parameter, ParameterList

class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

        self.mu = Parameter(torch.randn(3,5))
        self.y = ParameterList([Parameter(torch.randn(2,2)) for i in range(2)])

mod = Mod()
mod.state_dict()
list(mod.parameters())

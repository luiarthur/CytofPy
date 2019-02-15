import torch
from torch.nn import Parameter, ParameterList

class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

        self.mu = Parameter(torch.randn(3,5))
        self.y = ParameterList(Parameter(torch.randn(2,2)) for i in range(2))

# Test
mod = Mod()
mod.state_dict()
list(mod.parameters())

# Test
x = torch.randn(3,5)
x[2, 2] = torch.zeros(1, requires_grad=True)
loss = (torch.zeros(3,5) - x * 2.0).abs().sum()
loss.backward()
x.grad
x[2, 2].grad

x = torch.randn(3,5, requires_grad=True)
loss = (torch.zeros(3,5) - x * 2.0).abs().sum()
loss.backward()
x.grad

# Test
# https://discuss.pytorch.org/t/requires-gradient-on-only-part-of-a-matrix/5680
x = torch.randn(3,5, requires_grad=True)
loss = (torch.zeros(3,5) - x * 2.0).abs().sum()
loss.backward()
x.grad[2, 2] = 0.0
x.grad

# Test
# https://discuss.pytorch.org/t/requires-gradient-on-only-part-of-a-matrix/5680
x = torch.randn(3,5, requires_grad=True)
loss = (torch.zeros(3,5) - x[1] * 2.0).abs().sum()
loss.backward()
x.grad

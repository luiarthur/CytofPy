# I intend to use this as a Variational auto-encoder for the
# missing y.
# See paper: https://arxiv.org/abs/1312.6114

import torch

# Define sizes
input_size = 3
output_size = 2
hidden_size = 5

# Create multi-layer perceptron
fc1 = torch.nn.Linear(input_size, hidden_size)
act_fn = torch.nn.Tanh()
fc2 = torch.nn.Linear(hidden_size, output_size)

# Main
num_obs = 100
x = torch.randn(num_obs, input_size)
out = fc1(x)
out = act_fn(out)
out = fc2(out)
print(out)

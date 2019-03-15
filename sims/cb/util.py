import torch
import numpy as np

def get_one_hot(targets, nb_classes, use_np=False):
    if use_np:
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])
    else:
        # use torch
        res = torch.eye(nb_classes)[targets.clone().reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

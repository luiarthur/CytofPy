import unittest

import os
import torch
import math
import copy
import numpy as np
import pickle

import cytopy

class Test_ModelRuns(unittest.TestCase):
    def test_compiles(self):
        self.assertTrue(1 + 1 == 2)

        path_to_exp_results = 'results/test/'
        os.makedirs(path_to_exp_results, exist_ok=True)

        show_plots = False

        # torch.manual_seed(2) # Data with seed(2) is good
        torch.manual_seed(0) # This data is good
        np.random.seed(0)

        # TODO: Make this work
        data = cytopy.util.simdata(N=[300, 100, 200], L0=3, L1=3, J=8)

        y = copy.deepcopy(data['data']['y'])
        I = len(y)

        K = 10
        L = [2, 2]

        # model.debug=True
        priors = cytopy.model.default_priors(y, K=K, L=L)
        out = cytopy.model.fit(y, max_iter=10, lr_g=1e-1, lr_l=1e-2,
                               print_freq=1, eps=1e-6, priors=priors, minibatch_size=100,
                               tau=0.1, verbose=0, seed=1)

        # Save output
        with open('{}/out.p'.format(path_to_exp_results), 'wb') as f:
            pickle.dump(out, f)
            f.close()


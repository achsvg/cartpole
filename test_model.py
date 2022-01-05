import unittest

import numpy as np
import torch

from hlm import model
from hlm.test import test_base


class CartPolePoliciesTest(test_base.TestBase):

    def setUp(self):
        super().setUp()
        self.params = model.CartPolePoliciesParams(
            obs_size=100,
            num_actions=2,
        )
        self.policies = model.CartPolePolicies(self.params)

    def test_forward(self):
        # Check that it optimizes with a simple loss
        eps_len = 10
        observations = torch.randn(eps_len, self.params.obs_size)
        opt = torch.optim.AdamW(self.policies.parameters())
        target_index = 1
        
        # action should be 1 if observation 1 greater than 0.4, 0 otherwise
        targets = (observations[:,target_index] > .4).long()
        targets = torch.nn.functional.one_hot(targets, num_classes=2)

        for _ in range(5000):
            opt.zero_grad()
            actions = self.policies(observations)
            loss = torch.abs(targets - actions)
            
            # Stop iterations once we have a policy that outputs loss of 0.
            quit = False
            for j, l in enumerate(loss):
                if torch.abs(l).sum().item() == 0:
                    quit = True
                    break
            if quit:
                break

            loss = loss.mean()
            loss.backward()
            opt.step()

        # The chosen policy corresponds to observation j, as designed in the
        # target.
        self.assertTrue(quit)
        self.assertEqual(j, target_index)
        self.assertListEqual(actions[j].tolist(), targets.tolist())
        np.testing.assert_almost_equal(
            self.policies.variables[j].item(), 0.4, decimal=1)


if __name__ == '__main__':
    unittest.main()

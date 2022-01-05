from dataclasses import dataclass

import torch
from torch import nn
from torch.functional import Tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CartPolePoliciesParams:
    obs_size: int
    num_actions: int


class CartPolePolicies(nn.Module):
    """
    Cart pole observations are:
    [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    """

    def __init__(self, params: CartPolePoliciesParams):
        super().__init__()
        self.greater_than = GreaterThan(lr=1.0)
        variables = torch.zeros(params.obs_size, dtype=torch.float32)
        self.variables = nn.Parameter(variables).to(device)
        self._size = params.obs_size

    @property
    def size(self):
        return self._size

    def forward(self, observations: Tensor):
        """
        Outputs results of greater than operation between observations and 
        variables. Each of these operations can be used as a policy.
        """
        comp_pairs = self.greater_than(observations, self.variables)
        return comp_pairs


class CartPolePolicySelector(nn.Module):

    def __init__(self, num_policies: int):
        super().__init__()
        policy_attn = torch.ones(num_policies, dtype=torch.float32) * (1/num_policies)
        self._policy_attn = nn.Parameter(policy_attn).to(device)
        self._num_policies = num_policies

    @property
    def policy_attn(self):
        return torch.nn.functional.softmax(self._policy_attn, dim=-1)

    def forward(self, policies):
        """
        policies is num_policies x N x 2 if dim is 3, num_policies x 2 if dim 
        is 2
        """
        assert policies.size(dim=0) == self._num_policies
        if policies.dim() == 3:
            soft_policies = torch.einsum('ijk,i->ijk', policies, self._policy_attn)
        else:
            soft_policies = torch.einsum('ij,i->ij', policies, self._policy_attn)
        soft_policies = torch.nn.functional.softmax(soft_policies, dim=-1)
        return soft_policies


class GreaterThanFn(torch.autograd.Function):
    """
    Compare a and b, returns [0,1] if a < b and [1,0] else.
    """

    @staticmethod
    def forward(ctx, a, b, lr):
        ctx.set_materialize_grads(False)
        ctx.lr = lr
        compare = torch.where(a > b, 1, 0)
        result = torch.nn.functional.one_hot(compare.long(), num_classes=2).float()
        result.requires_grad = True
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = grad_output[..., 0] * -ctx.lr + grad_output[..., 1] * ctx.lr
        grad_b = grad_output[..., 0] * ctx.lr + grad_output[..., 1] * -ctx.lr
        return grad_a, grad_b, None


class GreaterThan(nn.Module):
    """ 
    Create a GreaterThan op for each pair of a and b.
    """

    def __init__(self, lr=0.1):
        super().__init__()
        self._lr = lr

    def forward(self, a, b):
        assert a.size(dim=-1) == b.size(dim=-1)
        compare_results = []
        for i in range(a.size(dim=-1)):
            compare_results.append(
                GreaterThanFn.apply(a[..., i], b[..., i], self._lr))
        return torch.stack(compare_results)  # size(a) * size(b), batch_size

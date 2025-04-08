from torch import Tensor
import torch
from phys_torch import grad, _div, curl, grad_and_value
from typing import reveal_type


def vfunc(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T
    return torch.stack((x.sin() * y * z, y.cos() * z * x, z.tan() * x * y)).T


def sfunc(inputs: Tensor) -> Tensor:
    return inputs.sin().sum(-1)


x = torch.randn((10, 3), requires_grad=True)
F = sfunc(x)

gradF = grad_and_value(sfunc)(x)
print(gradF[0], gradF[1])

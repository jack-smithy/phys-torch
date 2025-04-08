import phys_torch
import torch
from torch import Tensor
from itertools import product

###### helpers ######
### generate points


def grid(d):
    values = [list(p) for p in product((0, 0.5, 1, 1.5, 2), repeat=d)]
    return torch.tensor(values, requires_grad=True)


### scalar funcs


def scalar_func_2d_1(input: Tensor) -> Tensor:
    return input[..., 0] ** 2 + input[..., 1] ** 2


def scalar_func_2d_2(input: Tensor) -> Tensor:
    return input[..., 0] + input[..., 1]


### analytical gradients


def grad_scalar_func_2d_1(input: Tensor) -> Tensor:
    return torch.stack((2 * input[..., 0], 2 * input[..., 1])).T


def grad_scalar_func_2d_2(input: Tensor) -> Tensor:
    return torch.ones((input.shape[0], 2))


### vector funcs


def vector_func_3d3d(input: Tensor) -> Tensor:
    return torch.stack((input[..., 0] ** 2, input[..., 1] ** 2, input[..., 2] ** 2)).T


### analytical divergences


def div_vector_func_3d3d(input: Tensor) -> Tensor:
    return 2 * input.sum(dim=1)


###### tests ######


def test_installed():
    t = phys_torch.testies((2, 2))
    assert isinstance(t, Tensor)


def test_gradient_2d():
    x = grid(2)
    y = scalar_func_2d_1(x)

    grad_y = phys_torch.gradient(y, x)
    grad_y_analytical = grad_scalar_func_2d_1(x)

    assert torch.allclose(grad_y, grad_y_analytical)


def test_gradient_distributive_2d():
    x = grid(2)

    A = scalar_func_2d_1(x)
    B = scalar_func_2d_2(x)

    gradA = phys_torch.gradient(A, x)
    gradB = phys_torch.gradient(B, x)
    gradA_plus_gradB = gradA + gradB

    grad_AplusB = phys_torch.gradient(A + B, x)

    assert torch.allclose(grad_AplusB, gradA_plus_gradB)


def test_divergence_3d():
    x = grid(3)
    F = vector_func_3d3d(x)

    div_F = phys_torch.divergence(F, x)
    div_F_analytical = div_vector_func_3d3d(x)

    assert torch.allclose(div_F, div_F_analytical)


def test_curl_3d():
    x = grid(3)
    F = vector_func_3d3d(x)

    curl_F = phys_torch.curl(F, x)
    curl_F_analytical = torch.zeros_like(x)

    assert torch.allclose(curl_F, curl_F_analytical)

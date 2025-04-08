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


def scalar_func_2d_1(inputs: Tensor) -> Tensor:
    return inputs[..., 0] ** 2 + inputs[..., 1] ** 2


def scalar_func_2d_2(inputs: Tensor) -> Tensor:
    return inputs[..., 0] + inputs[..., 1]


### analytical gradients


def grad_scalar_func_2d_1(inputs: Tensor) -> Tensor:
    return torch.stack((2 * inputs[..., 0], 2 * inputs[..., 1])).T


def grad_scalar_func_2d_2(inputs: Tensor) -> Tensor:
    return torch.ones((inputs.shape[0], 2))


### vector funcs


def vector_func_3d3d_1(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T
    return torch.stack((x**2, y**2, z**2)).T


def vector_func_3d3d_2(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T
    return torch.stack((x.exp() * z.sin(), y**2 * z, 2 * x)).T


vector_funcs = [vector_func_3d3d_1, vector_func_3d3d_2]

### analytical divergences


def div_vector_func_3d3d_1(inputs: Tensor) -> Tensor:
    return 2 * inputs.sum(dim=1)


def div_vector_func_3d3d_2(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T
    return x.exp() * z.sin() + 2 * y * z


div_vector_funcs = [div_vector_func_3d3d_1, div_vector_func_3d3d_2]

### analytical curls


def curl_vector_func_3d3d_1(inputs: Tensor) -> Tensor:
    return torch.zeros_like(inputs)


def curl_vector_func_3d3d_2(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T
    return torch.stack((y**2, 2 - z.cos() * x.exp(), torch.zeros_like(z))).T


curl_vector_funcs = [curl_vector_func_3d3d_1, curl_vector_func_3d3d_2]

###### tests ######


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
    for func, div_func in zip(vector_funcs, div_vector_funcs):
        x = grid(3)
        F = func(x)

        div_F = phys_torch.divergence(F, x)
        div_F_analytical = div_func(x)

        assert torch.allclose(div_F, div_F_analytical)


def test_curl_3d():
    for func, curl_func in zip(vector_funcs, curl_vector_funcs):
        x = grid(3)
        F = func(x)

        curl_F = phys_torch.curl(F, x)
        curl_F_analytical = curl_func(x)

        assert torch.allclose(curl_F, curl_F_analytical)

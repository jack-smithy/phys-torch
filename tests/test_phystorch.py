import torch
from torch import Tensor
from itertools import product
from phys_torch import grad, grad_and_value, div, curl, check
import pytest

###### helpers ######
### generate points


def grid(d):
    values = [list(p) for p in product((0, 0.5, 1, 1.5, 2), repeat=d)]
    return torch.tensor(values, requires_grad=True)


### scalar funcs


def scalar_func_2d_1(inputs: Tensor) -> Tensor:
    return inputs[..., 0] ** 2 + inputs[..., 1] ** 2


def scalar_func_2d_2(inputs: Tensor) -> Tensor:
    return inputs[..., 0].sin() + inputs[..., 1].cos()


### analytical gradients


def grad_scalar_func_2d_1(inputs: Tensor) -> Tensor:
    return torch.stack((2 * inputs[..., 0], 2 * inputs[..., 1])).T


def grad_scalar_func_2d_2(inputs: Tensor) -> Tensor:
    return torch.stack((inputs[..., 0].cos(), -1 * inputs[..., 1].sin())).T


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
    grad_y = grad(scalar_func_2d_2)(x)
    grad_y_analytical = grad_scalar_func_2d_2(x)

    assert torch.allclose(grad_y, grad_y_analytical)


def test_gradient_distributive_2d():
    x = grid(2)

    gradA = grad(scalar_func_2d_1)(x)
    gradB = grad(scalar_func_2d_2)(x)
    gradA_plus_gradB = gradA + gradB

    grad_AplusB = grad(lambda x: scalar_func_2d_1(x) + scalar_func_2d_2(x))(x)

    assert torch.allclose(grad_AplusB, gradA_plus_gradB)


def test_grad_and_value_2d():
    x = grid(2)
    gradA, A = grad_and_value(scalar_func_2d_2)(x)

    gradA_analytical, A_analytical = grad_scalar_func_2d_2(x), scalar_func_2d_2(x)

    assert torch.allclose(gradA, gradA_analytical)
    assert torch.allclose(A, A_analytical)


def test_divergence_3d():
    for func, div_func in zip(vector_funcs, div_vector_funcs):
        x = grid(3)

        divF = div(func)(x)
        div_F_analytical = div_func(x)

        assert torch.allclose(divF, div_F_analytical)


def test_curl_3d():
    for func, curl_func in zip(vector_funcs, curl_vector_funcs):
        x = grid(3)

        curl_F = curl(func)(x)
        curl_F_analytical = curl_func(x)

        assert torch.allclose(curl_F, curl_F_analytical)


def test_scalar_func_args():
    x_good = grid(3)
    x_bad = torch.ones((1, 4))

    @check
    def func_pass(x: Tensor) -> Tensor:
        return x.sum(-1)

    @check
    def func_fail_output(x: Tensor) -> Tensor:
        return x

    @check
    def func_fail_input(x: Tensor):
        return x

    func_pass(x_good)

    with pytest.raises(ValueError):
        func_fail_output(x_good)

    with pytest.raises(ValueError):
        func_fail_input(x_bad)

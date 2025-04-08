import torch
from torch import Tensor
from typing import TypeAlias, Literal, Callable, Tuple

Dimension: TypeAlias = Literal["x", "y", "z"]
TensorFunc: TypeAlias = Callable[[Tensor], Tensor]
TensorFuncAux: TypeAlias = Callable[[Tensor], Tuple[Tensor, ...]]


def _grad_outputs(outputs: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    assert inputs.requires_grad and outputs.dim() == 1
    grad = torch.autograd.grad(
        outputs,
        inputs,
        torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return outputs, grad


def grad(func: TensorFunc) -> TensorFunc:
    return lambda x: _grad_outputs(func(x), x)[1]


def grad_and_value(func: TensorFunc) -> TensorFuncAux:
    return lambda x: tuple(reversed(_grad_outputs(func(x), x)))


def partial(
    outputs: Tensor,
    inputs: Tensor,
    output_dim: Dimension | int,
    input_dim: Dimension | int,
) -> Tensor:
    dim_map = {"x": 0, "y": 1, "z": 2}

    output_idx = output_dim if isinstance(output_dim, int) else dim_map[output_dim]
    input_idx = input_dim if isinstance(input_dim, int) else dim_map[input_dim]

    return torch.autograd.grad(
        outputs[:, output_idx],
        inputs,
        grad_outputs=torch.ones_like(outputs[:, output_idx]),
        retain_graph=True,
        create_graph=True,
    )[0][:, input_idx]


def _div(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad
    assert inputs.dim() == outputs.dim()
    assert outputs.shape[1] == 3 and inputs.shape[1] == 3

    dFx_dx = partial(outputs, inputs, "x", "x")
    dFy_dy = partial(outputs, inputs, "y", "y")
    dFz_dz = partial(outputs, inputs, "z", "z")

    return dFx_dx + dFy_dy + dFz_dz


def div(func: TensorFunc) -> TensorFunc:
    return lambda x: _div(func(x), x)


def _curl(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad
    assert inputs.dim() == outputs.dim() == 2
    assert outputs.shape[1] == 3 and inputs.shape[1] == 3

    dFy_dz = partial(outputs, inputs, "y", "z")
    dFz_dy = partial(outputs, inputs, "z", "y")

    dFz_dx = partial(outputs, inputs, "z", "x")
    dFx_dz = partial(outputs, inputs, "x", "z")

    dFx_dy = partial(outputs, inputs, "x", "y")
    dFy_dx = partial(outputs, inputs, "y", "x")

    curl = torch.zeros(
        (outputs.shape[0], 3),
        dtype=outputs.dtype,
        device=outputs.device,
    )

    curl[:, 0] = dFy_dz - dFz_dy
    curl[:, 1] = dFz_dx - dFx_dz
    curl[:, 2] = dFx_dy - dFy_dx

    return curl


def curl(func: TensorFunc) -> TensorFunc:
    return lambda x: _curl(func(x), x)

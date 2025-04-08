import torch
from torch import Tensor
from typing import Callable, TypeAlias, Literal

TensorFunc: TypeAlias = Callable[[Tensor], Tensor]
Dimension: TypeAlias = Literal["x", "y", "z"]


def testies(shape):
    return torch.rand(shape)


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


def gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad
    assert outputs.dim() == 1

    grad_outputs = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]

    return grad_outputs


def divergence(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad
    assert inputs.dim() == outputs.dim()

    # ∂Fx/∂x
    dFx_dx = partial(outputs, inputs, "x", "x")

    # ∂Fx/∂x
    dFy_dy = partial(outputs, inputs, "y", "y")

    # ∂Fx/∂x
    dFz_dz = partial(outputs, inputs, "z", "z")

    return dFx_dx + dFy_dy + dFz_dz


def curl(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad
    assert inputs.dim() == outputs.dim() == 2
    assert outputs.shape[1] == 3 and inputs.shape[1] == 3

    # ∂Fy/∂z - ∂Fz/∂y
    dFy_dz = partial(outputs, inputs, "y", "z")
    dFz_dy = partial(outputs, inputs, "z", "y")

    # ∂Fz/∂x - ∂Fx/∂z
    dFz_dx = partial(outputs, inputs, "z", "x")
    dFx_dz = partial(outputs, inputs, "x", "z")

    # ∂Fx/∂y - ∂Fy/∂x
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

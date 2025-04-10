import torch
from torch import Tensor
from typing import TypeAlias, Literal, Callable, Tuple, overload

Dimension: TypeAlias = Literal["x", "y", "z"]

_TensorFunc: TypeAlias = Callable[[Tensor], Tensor]
_TensorFuncAux: TypeAlias = Callable[[Tensor], Tuple[Tensor, Tensor]]


def _grad_outputs(outputs: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculates the gradient of `outputs` with respect to the `inputs` tensor.
    `create_graph` and `retain_graph` are both set to true to allow for higher order derivatives.

    Args:
        outputs (Tensor): Tensor result from a computation.
        inputs (Tensor): The input tensor to the computation to differentiate with respect to.

    Returns:
        Tuple[Tensor, Tensor]: Gradents at the output points, and the output value
    """
    assert inputs.requires_grad and outputs.requires_grad

    if inputs.dim() != 2:
        raise ValueError(
            "Input function must accept tensor with shape (batch_size, dim)"
        )

    if outputs.dim() != 1:
        raise ValueError("Output tensor must be shape (batch_size,)")

    grad = torch.autograd.grad(
        outputs,
        inputs,
        torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

    return grad, outputs


@overload
def grad(func: _TensorFunc, return_value: Literal[False] = ...) -> _TensorFunc: ...


@overload
def grad(func: _TensorFunc, return_value: Literal[True]) -> _TensorFuncAux: ...


def grad(func: _TensorFunc, return_value: bool = False) -> _TensorFunc | _TensorFuncAux:
    """
    grad operator computing gradients of func with respect to the input for functions `R^dim->R`.
    This operator can be composed to compute higher-order gradients.

    The input function should accept a batched tensor of shape `(batch_size, dim)` and return
    a tensor of shape `(batch_size, )`. The gradient computation is performed for each
    batch element.

    ```python
    x = torch.randn((4, 3), requires_grad=True)  # 4 points in 3D space

    def func(x):
        return x[:, 0] ** 2 + x[:, 1] * x[:, 2]

    gradF = grad(func)(x)
    gradF, F = grad(func, return_value=True)(x)
    ```

    Args:
        func (_TensorFunc): A function that takes a tensor of shape `(n, 3)`
                            and returns a tensor of shape `(n,)`.
        return_value (bool): Whether to return the values of the function at the points.

    Returns:
        _TensorFunc: A function that computes the gradient of the input function
                     with respect to its input.
    """
    if return_value:
        return lambda x: _grad_outputs(func(x), x)

    return lambda x: _grad_outputs(func(x), x)[0]


def _partial(
    outputs: Tensor,
    inputs: Tensor,
    output_dim: Dimension | int,
    input_dim: Dimension | int,
) -> Tensor:
    """
    Calculates the (input_dim, output_dim) partial derivative of outputs with respect to inputs.

    Args:
        outputs (Tensor): Tensor result from a computation.
        inputs (Tensor): The input tensor to the computation to differentiate with respect to.
        output_dim (Dimension | int):
        input_dim (Dimension | int):

    Returns:
        Tensor: _description_
    """
    assert inputs.requires_grad

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


def partial(
    func: _TensorFunc,
    input_dim: Dimension,
    output_dim: Dimension,
) -> _TensorFunc:
    """
    Operator which calculates the (input_dim, output_dim) partial derivative of outputs with respect to inputs.

    For a function `R^n->R^m`, it calculates `∂F_(output_dim)/∂_(input_dim)`, where `input_dim ∈ {0,...,n-1}`, and `output_dim ∈ {0,...,m-1}`.
    If the function acts `R^3->R^3`, the letters "x", "y", and "z" can be used to denote the dimensions for the derivative.

    ```python
    inputs = torch.randn((4, 3))

    def func(x):
        return torch.stack((x[:, 0] ** 2 + x[:, 1], x[:, 2], x[:, 0].sin())).T

    df_y/dx = partial(func, input_dim="x", output_dim="y")(inputs) # calculates ∂F_y/∂x
    df_z/dy = partial(func, input_dim=1, output_dim=2)(inputs) # calculates ∂F_z/∂y
    ```

    Args:
        outputs (Tensor): Tensor result from a computation.
        inputs (Tensor): The input tensor to the computation to differentiate with respect to.
        output_dim (Dimension | int):
        input_dim (Dimension | int): _description_

    Returns:
        Tensor: _description_
    """

    return lambda x: _partial(
        outputs=func(x),
        inputs=x,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _div_outputs(outputs: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculates the divergence of outputs with respect to inputs.

    Args:
        outputs (Tensor): _description_
        inputs (Tensor): _description_

    Returns:
        Tuple[Tensor, Tensor]: _description_
    """
    assert inputs.requires_grad and outputs.requires_grad
    assert inputs.dim() == outputs.dim()
    assert outputs.shape[1] == inputs.shape[1]

    divs = torch.empty((inputs.shape[0]))
    for i in range(inputs.shape[1]):
        divs += _partial(outputs, inputs, i, i)

    return divs, outputs


@overload
def div(func: _TensorFunc, return_value: Literal[False] = ...) -> _TensorFunc: ...


@overload
def div(func: _TensorFunc, return_value: Literal[True]) -> _TensorFuncAux: ...


def div(func: _TensorFunc, return_value: bool = False) -> _TensorFunc | _TensorFuncAux:
    """_summary_

    Args:
        func (_TensorFunc): _description_
        return_value (bool, optional): _description_. Defaults to False.

    Returns:
        _TensorFunc | _TensorFuncAux: _description_
    """
    if return_value:
        return lambda x: _div_outputs(outputs=func(x), inputs=x)

    return lambda x: _div_outputs(outputs=func(x), inputs=x)[0]


def _curl_outputs(outputs: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    """_summary_

    Args:
        outputs (Tensor): _description_
        inputs (Tensor): _description_

    Returns:
        Tuple[Tensor, Tensor]: _description_
    """
    assert inputs.requires_grad
    assert inputs.dim() == outputs.dim() == 2
    assert outputs.shape[1] == 3 and inputs.shape[1] == 3

    dFy_dz = _partial(outputs, inputs, "y", "z")
    dFz_dy = _partial(outputs, inputs, "z", "y")

    dFz_dx = _partial(outputs, inputs, "z", "x")
    dFx_dz = _partial(outputs, inputs, "x", "z")

    dFx_dy = _partial(outputs, inputs, "x", "y")
    dFy_dx = _partial(outputs, inputs, "y", "x")

    curl = torch.zeros(
        (outputs.shape[0], 3),
        dtype=outputs.dtype,
        device=outputs.device,
    )

    curl[:, 0] = dFy_dz - dFz_dy
    curl[:, 1] = dFz_dx - dFx_dz
    curl[:, 2] = dFx_dy - dFy_dx

    return curl, outputs


@overload
def curl(func: _TensorFunc, return_value: Literal[False] = ...) -> _TensorFunc: ...


@overload
def curl(func: _TensorFunc, return_value: Literal[True]) -> _TensorFuncAux: ...


def curl(func: _TensorFunc, return_value: bool = False) -> _TensorFunc | _TensorFuncAux:
    """_summary_

    Args:
        func (_TensorFunc): _description_
        return_value (bool, optional): _description_. Defaults to False.

    Returns:
        _TensorFunc | _TensorFuncAux: _description_
    """
    if return_value:
        return lambda x: _curl_outputs(outputs=func(x), inputs=x)

    return lambda x: _curl_outputs(outputs=func(x), inputs=x)[0]

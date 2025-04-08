import torch
from torch import Tensor
from typing import TypeAlias, Literal, Callable, Tuple
import functools

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
    assert inputs.requires_grad

    grad = torch.autograd.grad(
        outputs,
        inputs,
        torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

    return grad, outputs


def grad(func: _TensorFunc) -> _TensorFunc:
    """
    grad operator computing gradients of func with respect to the input for functions `R^3->R`.
    This operator can be nested to compute higher-order gradients.

    The input function should accept a batched tensor of shape `(n, 3)` and return
    a tensor of shape `(n,)`. The gradient computation is performed for each
    batch element.


    ```python
    import phys_torch
    import torch

    x = torch.randn((4, 3), requires_grad=True)  # 4 points in 3D space

    def Ffunc(x):
        return x[:, 0] ** 2 + x[:, 1] * x[:, 2]

    gradF = phys_torch.grad(Ffunc)(x)
    ```

    Args:
        func (_TensorFunc): A function that takes a tensor of shape `(n, 3)`
                            and returns a tensor of shape `(n,)`.

    Returns:
        _TensorFunc: A function that computes the gradient of the input function
                     with respect to its input.
    """

    def wrapper(x: Tensor):
        if x.shape[1] != 3 and x.dim() != 2:
            raise ValueError("Input function must accept tensor with shape (n ,3)")

        y = func(x)

        if y.dim() != 1:
            raise ValueError("Output tensor must be shape (n,)")

        return _grad_outputs(y, x)[0]

    return wrapper


def grad_and_value(func: _TensorFunc) -> _TensorFuncAux:
    """
    grad operator computing gradients and values of func with respect to the input for functions `R^3->R`.
    This operator can be nested to compute higher-order gradients.

    The input function should accept a batched tensor of shape `(n, 3)` and return
    a tensor of shape `(n,)`. The gradient computation is performed for each
    batch element.


    ```python
    import phys_torch
    import torch

    x = torch.randn((4, 3), requires_grad=True)  # 4 points in 3D space

    def Ffunc(x):
        return x[:, 0] ** 2 + x[:, 1] * x[:, 2]

    F, gradF = phys_torch.grad_and_value(Ffunc)(x)
    ```

    Args:
        func (_TensorFunc): A function that takes a tensor of shape `(n, 3)`
            and returns a tensor of shape `(n,)`.

    Returns:
        _TensorFuncAux: A function that computes the gradient of the input function
            with respect to its input, and the values of the function at those points.
    """

    def wrapper(x: Tensor):
        if x.shape[1] != 3 and x.dim() != 2:
            raise ValueError("Input function must accept tensor with shape (n ,3)")

        y = func(x)

        if y.dim() != 1:
            raise ValueError("Output tensor must be shape (n,)")

        return _grad_outputs(y, x)

    return wrapper


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

    For a function `R^3->R^3`, it calculates `∂F_(output_dim)/∂_(input_dim)`

    Args:
        outputs (Tensor): Tensor result from a computation.
        inputs (Tensor): The input tensor to the computation to differentiate with respect to.
        output_dim (Dimension | int):
        input_dim (Dimension | int): _description_

    Returns:
        Tensor: _description_
    """

    def wrapper(x):
        y = func(x)
        return _partial(y, x, input_dim=input_dim, output_dim=output_dim)

    return wrapper


def _div(outputs: Tensor, inputs: Tensor) -> Tensor:
    assert inputs.requires_grad, "The input tensor must have `requires_grad=True`"
    assert inputs.dim() == outputs.dim()
    assert outputs.shape[1] == 3 and inputs.shape[1] == 3

    dFx_dx = _partial(outputs, inputs, "x", "x")
    dFy_dy = _partial(outputs, inputs, "y", "y")
    dFz_dz = _partial(outputs, inputs, "z", "z")

    return dFx_dx + dFy_dy + dFz_dz


def div(func: _TensorFunc) -> _TensorFunc:
    return lambda x: _div(func(x), x)


def _curl(outputs: Tensor, inputs: Tensor) -> Tensor:
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

    return curl


def curl(func: _TensorFunc) -> _TensorFunc:
    return lambda x: _curl(func(x), x)


def check(func):
    """
    Checks a function accepts and returns valid shapes for scalar functions

    Args:
        func (_type_): _description_

    Raises:
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        # Check input shape
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor as input, got {type(x)}")

        if len(x.shape) != 2 or x.shape[1] != 3:
            raise ValueError(f"Input tensor must have shape (n, 3), got {x.shape}")

        # Call the original function
        result = func(x, *args, **kwargs)

        # Check output shape
        if not isinstance(result, Tensor):
            raise TypeError(f"Expected Tensor as output, got {type(result)}")

        if len(result.shape) != 1 or result.shape[0] != x.shape[0]:
            raise ValueError(
                f"Output tensor must have shape ({x.shape[0]},), got {result.shape}"
            )

        return result

    return wrapper

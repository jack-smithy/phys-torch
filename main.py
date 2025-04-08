from torch import Tensor
import torch


def partials_1(inputs: Tensor) -> Tensor:
    x, y, z = inputs.T

    grad_Fx = torch.stack(
        (
            z.sin() * z.exp(),
            torch.zeros_like(y),
            z.cos() * x.exp(),
        )
    ).T

    grad_Fy = torch.stack(
        (
            torch.zeros_like(x),
            2 * y * z,
            y**2.0,
        )
    ).T

    grad_Fz = torch.stack(
        (
            2 * torch.ones_like(x),
            torch.zeros_like(y),
            torch.zeros_like(z),
        )
    ).T

    return torch.stack((grad_Fx, grad_Fy, grad_Fz)).swapaxes(0, 1)


inputs = torch.ones((10, 3))

partials = partials_1(inputs)

# .diagonal(0, 1, 2).sum(1)

print(partials)

import phys_torch
import torch

x = torch.randn((4, 3), requires_grad=True)  # 4 points in 3D space


@phys_torch.check
def F(x):
    return x


gradF = phys_torch.grad(F)(x)

print(gradF)

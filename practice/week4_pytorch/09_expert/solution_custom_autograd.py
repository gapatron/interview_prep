"""Solution: Custom LeakyReLU with autograd.Function; save input and neg_slope for backward."""

import torch
from torch.autograd import Function


class LeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, neg_slope: float = 0.01) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.neg_slope = neg_slope
        output = torch.where(input >= 0, input, neg_slope * input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        neg_slope = ctx.neg_slope
        mask = (input >= 0).float() + (input < 0).float() * neg_slope
        return grad_output * mask, None


def leaky_relu_custom(input: torch.Tensor, neg_slope: float = 0.01) -> torch.Tensor:
    return LeakyReLUFunction.apply(input, neg_slope)


if __name__ == "__main__":
    x = torch.randn(2, 4, requires_grad=True)
    y = leaky_relu_custom(x, neg_slope=0.1)
    y.sum().backward()
    assert x.grad is not None
    x2 = x.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.leaky_relu(x2, 0.1)
    y2.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)
    print("Custom LeakyReLU OK.")

"""
Level 5 / Expert — Custom autograd.Function
===========================================
Implement a custom backward for a simple operation (e.g. LeakyReLU or scale+clamp)
using torch.autograd.Function. Industry: custom kernels, fused ops, quantized backward.
  - Define forward(ctx, input, ...) and backward(ctx, grad_output)
  - Save tensors needed in backward with ctx.save_for_backward(...) or ctx.attr = value
  - In backward: grad_input = grad_output * (mask or derivative)
Implement LeakyReLU: forward = x if x >= 0 else neg_slope * x; backward = grad_output * (1 or neg_slope).
"""

import torch
from torch.autograd import Function


class LeakyReLUFunction(Function):
    """Custom LeakyReLU: forward x if x>=0 else neg_slope*x; backward grad * (1 or neg_slope)."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, neg_slope: float = 0.01) -> torch.Tensor:
        # TODO: ctx.save_for_backward(...) for tensors; ctx.neg_slope = neg_slope
        # TODO: output = input.clamp(min=0) + neg_slope * input.clamp(max=0); return output
        pass

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: (input,) = ctx.saved_tensors; neg_slope = ctx.neg_slope
        # TODO: mask = (input >= 0).float() + (input < 0).float() * neg_slope
        # TODO: return grad_output * mask, None  # None for neg_slope (non-tensor)
        pass


def leaky_relu_custom(input: torch.Tensor, neg_slope: float = 0.01) -> torch.Tensor:
    return LeakyReLUFunction.apply(input, neg_slope)


if __name__ == "__main__":
    x = torch.randn(2, 4, requires_grad=True)
    y = leaky_relu_custom(x, neg_slope=0.1)
    y.sum().backward()
    assert x.grad is not None
    # Compare with built-in (same input)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.leaky_relu(x2, 0.1)
    y2.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)
    print("Custom LeakyReLU OK.")

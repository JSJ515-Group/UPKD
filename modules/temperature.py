"""Learnable temperature module for knowledge distillation."""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverse the incoming gradient by a configurable coefficient."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = grads.new_tensor(ctx.lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class LearnableTemperature(nn.Module):
    """Learn a bounded temperature using gradient reversal."""

    def __init__(self, t_start=1.0, t_end=20.0):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end
        self.temperature_parameter = nn.Parameter(
            torch.ones(1),
            requires_grad=True,
        )
        self.grl = GradientReversal()

    def forward(self, logit_s, logit_t, lambda_=1.0):
        """Return the temperature value used by the distillation loss.

        Args:
            logit_s: Student logits retained for training-loop compatibility.
            logit_t: Teacher logits retained for training-loop compatibility.
            lambda_: Gradient reversal coefficient.

        Returns:
            The bounded learnable temperature.
        """
        temp = self.grl(self.temperature_parameter, lambda_)
        temp = self.t_start + (
            self.t_end - self.t_start
        ) * torch.sigmoid(temp)
        return temp

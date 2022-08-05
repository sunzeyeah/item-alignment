
import torch

from torch.nn.modules.loss import _Loss


class EuclideanDistanceLoss(_Loss):
    r"""Measures the loss given an input tensor :math:`x` (euclidean distance) and a labels tensor :math:`y`
    (containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n^{y_n}
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation
          operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input
    """
    __constants__ = ['reduction']
    # margin: float

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(EuclideanDistanceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.pow(input, target)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss


class HingeLoss(_Loss):
    r"""Measures the loss given an input tensor :math:`x` (inner product) and a labels tensor :math:`y`
    (containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            \max \{0, \Delta - y_n * x_n\}
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation
          operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 1.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(HingeLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, device=input.device)
        loss = torch.max(zero, self.margin - input * target)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss

import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss

def kpt_l1_loss(pred, target, vis_flag, weight=None, reduction='mean', avg_factor=None):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    # import ipdb; ipdb.set_trace()
    loss = loss * vis_flag
    loss = loss.sum(dim=1)
    
    if weight is not None:
        assert loss.shape == weight.shape
        loss = loss * weight
    
    if reduction == 'mean' and avg_factor is not None:
        loss = loss.sum() / avg_factor
    
    return loss


@LOSSES.register_module()
class KptL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KptL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                vis_flag,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import ipdb; ipdb.set_trace()
        loss_bbox = self.loss_weight * kpt_l1_loss(
            pred,
            target,
            vis_flag, 
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox
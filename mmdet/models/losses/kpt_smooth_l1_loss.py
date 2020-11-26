import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


def kpt_smooth_l1_loss(pred, target, vis_flag, weight, beta, reduction='mean', avg_factor=None):
    assert beta > 0
    assert pred.size() == target.size() == vis_flag.size() and target.numel() > 0
    # import ipdb; ipdb.set_trace()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    loss = loss * vis_flag
    loss = loss.sum(dim=1)
    
    if weight is not None:
        assert loss.shape == weight.shape
        loss = loss * weight
    # else:
        # raise NotImplementedError
    if reduction == 'mean' and avg_factor is not None:
        loss = loss.sum() / avg_factor
    else:
        raise NotImplementedError

    return loss

@LOSSES.register_module
class KptSmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(KptSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                vis_flag,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * kpt_smooth_l1_loss(
            pred,
            target,
            vis_flag, 
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss
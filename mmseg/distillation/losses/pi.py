import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class PixelWiseLoss(nn.Module):

    def __init__(self, name, upsample=False, tau=1.0, loss_weight=10.0):
        super(PixelWiseLoss, self).__init__()
        self.name = name
        self.upsample = upsample
        self.tau = tau
        self.loss_weight = loss_weight
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_S, preds_T):
        h, w = preds_T.size(2), preds_T.size(3)
        if self.upsample:
            scale_pred = F.upsample(input=preds_S, size=(h * 8, w * 8), mode='bilinear', align_corners=True)
            scale_soft = F.upsample(input=preds_T, size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds_S
            scale_soft = preds_T
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.tau, dim=1),
                                 F.softmax(scale_soft / self.tau, dim=1))
        return self.loss_weight * loss


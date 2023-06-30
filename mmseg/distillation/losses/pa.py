import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class PairWiseLoss(nn.Module):
    def __init__(self, name, scale=0.5, loss_weight=1.0):
        super(PairWiseLoss, self).__init__()
        self.name = name
        self.scale = scale
        self.loss_weight = loss_weight
        self.criterion = sim_dis_compute

    def forward(self, feat_S, feat_T):
        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                               ceil_mode=True)  # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))

        return self.loss_weight * loss


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class CIRKDLoss(nn.Module):

    def __init__(self, name, tau=0.1, pooling=False, loss_weight=1.0):
        super(CIRKDLoss, self).__init__()
        self.name = name
        self.tau = tau
        self.pooling = pooling
        self.loss_weight = loss_weight

    def forward(self, feat_S, feat_T):
        # feat_T = self.concat_all_gather(feat_T)
        # feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        if self.pooling:
            avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
            feat_S = avg_pool(feat_S)
            feat_T = avg_pool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        sim_dis = torch.tensor(0.).cuda()
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.tau, dim=1)
                p_t = F.softmax(t_sim_map / self.tau, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return self.loss_weight * sim_dis

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1


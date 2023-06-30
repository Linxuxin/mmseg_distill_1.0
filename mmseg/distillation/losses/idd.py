import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class InterClassLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, name, distance_type='euclidean', student_channels=19, teacher_channels=19,
                 loss_weight=0.5, sigma=1.0, tau=4.0, loss_type='kl', num_classes=19, decoupled=True):
        super(InterClassLoss, self).__init__()
        self.name = name
        self.distance_type = distance_type
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.tau = tau
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.ignore_label = 255
        self.decoupled = decoupled

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.embed_s = Embed(student_channels, student_channels)
        self.embed_t = Embed(teacher_channels, student_channels)

    def forward(self, preds_S, preds_T, gt):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape
        gt = F.interpolate(gt.float(), size=[H, W], mode='bilinear', align_corners=False).int()

        if self.distance_type == 'euclidean':
            loss = self.euclidean_distance(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'gaussian':
            loss = self.gaussian_distance(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'contrastive':
            loss = self.contrastive_learning(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'cosine':
            loss = self.cosine_distance(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'cross_sim':
            loss = self.cross_similarity(preds_S, preds_T, gt, self.loss_type)

        return self.loss_weight * loss

    def euclidean_distance(self, preds_S, preds_T, gt, loss_type):
        loss = 0
        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            vi_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
            vi_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

            for j in range(self.num_classes):
                if i < j:
                    mask_feat_S = (gt == j).float()
                    mask_feat_T = (gt == j).float()
                    vj_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
                    vj_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

                    e_ij_S = F.pairwise_distance(vi_S, vj_S, p=2)
                    e_ij_T = F.pairwise_distance(vi_T, vj_T, p=2)
                    loss += 0.5 * (e_ij_T - e_ij_S) ** 2

        loss = loss.mean()

        return loss

    def gaussian_distance(self, preds_S, preds_T, gt, loss_type):
        N, C, H, W = preds_S.shape
        loss = 0
        g_d_S = []
        g_d_T = []
        preds_S = F.normalize(preds_S, p=2, dim=1)
        preds_T = F.normalize(preds_T, p=2, dim=1)

        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            vi_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
            vi_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

            for j in range(self.num_classes):
                if i < j:
                    mask_feat_S = (gt == j).float()
                    mask_feat_T = (gt == j).float()
                    vj_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
                    vj_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

                    e_ij_S = F.pairwise_distance(vi_S, vj_S, p=2)
                    e_ij_T = F.pairwise_distance(vi_T, vj_T, p=2)
                    g_d_S.append(torch.exp(-e_ij_S ** 2 / (2 * self.sigma ** 2)))
                    g_d_T.append(torch.exp(-e_ij_T ** 2 / (2 * self.sigma ** 2)))
                    # g_d_S.append(torch.exp(-(vi_S - vj_S) ** 2 / (2 * self.sigma ** 2)))
                    # g_d_T.append(torch.exp(-e_ij_T ** 2 / (2 * self.sigma ** 2)))
                    # loss += 0.5 * (e_ij_T - e_ij_S) ** 2

        # gau_S = torch.stack(g_d_S) / sum(torch.stack(g_d_S))
        # gau_T = torch.stack(g_d_T) / sum(torch.stack(g_d_T))
        gau_S = torch.stack(g_d_S)
        gau_T = torch.stack(g_d_T)

        if loss_type == 'kl':
            softmax_pred_T = F.softmax(gau_T / self.tau, dim=0)
            logsoftmax = torch.nn.LogSoftmax(dim=0)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(gau_T / self.tau) -
                             softmax_pred_T *
                             logsoftmax(gau_S / self.tau)) * (self.tau ** 2)
        elif loss_type == 'l2':
            loss_mse = nn.MSELoss(reduction='sum')
            loss = loss_mse(gau_S, gau_T)

        loss = loss / N

        return loss

    def cosine_distance(self, preds_S, preds_T, gt, loss_type):
        preds_S = self.embed_s(preds_S)
        preds_T = self.embed_s(preds_T)

        N, C, H, W = preds_S.shape
        loss_rec = 0
        list_S = []
        list_T = []
        cos = nn.CosineSimilarity(dim=1)

        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            vi_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
            vi_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

            loss_rec += cos(vi_S, vi_T)

            list_S.append(cos(preds_S, vi_S.unsqueeze(2).unsqueeze(3))/self.tau)
            list_T.append(cos(preds_T, vi_T.unsqueeze(2).unsqueeze(3))/self.tau)

        p_S = torch.stack(list_S, dim=1) / torch.stack(list_S, dim=1).sum(1).unsqueeze(1)
        p_T = torch.stack(list_T, dim=1) / torch.stack(list_T, dim=1).sum(1).unsqueeze(1)

        softmax_pred_T = F.softmax(p_T.view(-1, H * W), dim=0)
        logsoftmax = torch.nn.LogSoftmax(dim=0)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(p_T.view(-1, H * W)) -
                         softmax_pred_T *
                         logsoftmax(p_S.view(-1, H * W))) / (N * H * W)

        loss_rec = 1 - torch.sum(loss_rec) / (N * self.num_classes)

        return loss + loss_rec

    def contrastive_learning(self, preds_S, preds_T, gt, loss_type):
        N, C, H, W = preds_S.shape
        loss = 0

        preds_S = nn.functional.normalize(preds_S, p=2, dim=1)
        preds_T = nn.functional.normalize(preds_T, p=2, dim=1)
        # preds_S_flatten = preds_S.view(N, C, -1)
        # preds_T_flatten = preds_T.view(N, C, -1)
        #
        # logits = torch.bmm(preds_S_flatten.transpose(1, 2), preds_T_flatten)
        # logits = (torch.clamp(logits, min=-1, max=1)) / self.tau

        mask = torch.eq(gt.view(N, 1, -1).transpose(1, 2), gt.view(N, 1, -1)).float()

        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()

            pos_pixel_S = mask_feat_S * preds_S
            pos_pixel_T = mask_feat_T * preds_T
            neg_pixel_T = (1 - mask_feat_T) * preds_T

            dot_pos = torch.bmm(pos_pixel_S.view(N, C, -1).transpose(1, 2), pos_pixel_T.view(N, C, -1))
            dot_neg = torch.bmm(pos_pixel_S.view(N, C, -1).transpose(1, 2), neg_pixel_T.view(N, C, -1))

            dot_pos -= torch.max(dot_pos, dim=2, keepdim=True)[0]
            dot_neg -= torch.max(dot_neg, dim=2, keepdim=True)[0]

            cor_pos = torch.exp(dot_pos / self.tau)
            cor_neg = mask * torch.exp(dot_neg / self.tau)

            loss += -torch.log(cor_pos/(cor_pos+cor_neg+1e-6))

        loss = torch.mean(loss)

        return loss

    def cross_similarity(self, preds_S, preds_T, gt, loss_type):
        N, C, H, W = preds_T.shape
        loss = 0
        cos = nn.CosineSimilarity(dim=1)

        # for bs in range(N):
        #     this_feature_S = preds_S[bs].contiguous().view(C, -1)
        #     this_feature_T = preds_T[bs].contiguous().view(C, -1)
        #
        #     this_label = gt[bs].contiguous().view(-1)
        #     this_label_ids = torch.unique(this_label)
        #     this_label_ids = [x for x in this_label_ids if x != self.ignore_label]
        #
        #     if len(this_label_ids) == 0:
        #         continue
        #
        #     for lb in this_label_ids:
        #         idx_gt_mask = (this_label == lb).nonzero()
        #         idx_other_mask = (this_label != lb).nonzero()
        #
        #         gt_mask_feat_S = this_feature_S[:, idx_gt_mask].squeeze(2)
        #         gt_mask_feat_T = this_feature_T[:, idx_gt_mask].squeeze(2)
        #         other_mask_feat_S = this_feature_S[:, idx_other_mask].squeeze(2)
        #         other_mask_feat_T = this_feature_T[:, idx_other_mask].squeeze(2)
        #
        #         vi_S = torch.mean(gt_mask_feat_S, dim=1)
        #         vi_T = torch.mean(gt_mask_feat_T, dim=1)
        #
        #         cross_sim_other_ST = cos(vi_S.unsqueeze(1), other_mask_feat_T)
        #         sim_other_SS = cos(vi_S.unsqueeze(1), other_mask_feat_S)
        #
        #         cross_sim_other_TS = cos(vi_T.unsqueeze(1), other_mask_feat_S)
        #         sim_other_TT = cos(vi_T.unsqueeze(1), other_mask_feat_T)
        #
        #         cross_sim_gt_ST = cos(vi_S.unsqueeze(1), gt_mask_feat_T)
        #         sim_gt_SS = cos(vi_S.unsqueeze(1), gt_mask_feat_S)
        #
        #         cross_sim_gt_TS = cos(vi_T.unsqueeze(1), gt_mask_feat_S)
        #         sim_gt_TT = cos(vi_T.unsqueeze(1), gt_mask_feat_T)
        #
        #         loss += (self.contrast_sim_kd(sim_other_SS, cross_sim_other_ST, dim=0, reduction='sum') +
        #                  self.contrast_sim_kd(sim_other_TT, cross_sim_other_TS, dim=0, reduction='sum') +
        #                  self.contrast_sim_kd(sim_gt_SS, cross_sim_gt_ST, dim=0, reduction='sum') +
        #                  self.contrast_sim_kd(sim_gt_TT, cross_sim_gt_TS, dim=0, reduction='sum'))

        for i in range(self.num_classes):
            # if i not in gt:
            #     continue
            gt_mask = (gt == i).float()
            other_mask = (gt != i).float()

            gt_mask_feat_S = gt_mask * preds_S
            gt_mask_feat_T = gt_mask * preds_T
            other_mask_feat_S = other_mask * preds_S
            other_mask_feat_T = other_mask * preds_T

            vi_S = gt_mask_feat_S.sum(-1).sum(-1) / (gt_mask.sum(-1).sum(-1) + 1e-6)
            vi_T = gt_mask_feat_T.sum(-1).sum(-1) / (gt_mask.sum(-1).sum(-1) + 1e-6)

            # for j in range(self.num_classes):
            #     if i < j:
            #         gt_mask_j = (gt == j).float()
            #         vj_S = (gt_mask_j * preds_S).sum(-1).sum(-1) / (gt_mask_j.sum(-1).sum(-1) + 1e-6)
            #         vj_T = (gt_mask_j * preds_T).sum(-1).sum(-1) / (gt_mask_j.sum(-1).sum(-1) + 1e-6)
            #
            #         e_ij_SS = F.pairwise_distance(vi_S, vj_S, p=2)
            #         e_ij_ST = F.pairwise_distance(vi_S, vj_T, p=2)
            #         e_ij_TS = F.pairwise_distance(vi_T, vj_S, p=2)
            #         e_ij_TT = F.pairwise_distance(vi_T, vj_T, p=2)
            #         loss += (0.001 * (0.25 * ((e_ij_ST - e_ij_SS) ** 2 + (e_ij_TS - e_ij_TT) ** 2))).mean()

            # cross_sim_gt_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_T)
            # sim_gt_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)
            #
            # cross_sim_gt_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)
            # sim_gt_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_T)
            #
            # loss += (self.contrast_sim_kd(sim_gt_SS, cross_sim_gt_ST) +
            #          self.contrast_sim_kd(sim_gt_TT, cross_sim_gt_TS)) / 2

            if self.decoupled:
                cross_sim_other_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), other_mask_feat_T)
                sim_other_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), other_mask_feat_S)

                cross_sim_other_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), other_mask_feat_S)
                sim_other_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), other_mask_feat_T)

                # cross_kl_anchor = self.contrast_sim_kd(vi_S, vi_T)

                # OOM
                cross_sim_gt_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_T)
                sim_gt_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)

                cross_sim_gt_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)
                sim_gt_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_T)

                # loss += (self.contrast_sim_kd(cross_sim_other_TS, sim_other_TT) +
                #          self.contrast_sim_kd(cross_sim_gt_TS, sim_gt_TT)) / 2

                loss += (self.contrast_sim_kd(sim_other_SS, cross_sim_other_ST) +
                         self.contrast_sim_kd(cross_sim_other_TS, sim_other_TT) +
                         self.contrast_sim_kd(sim_gt_SS, cross_sim_gt_ST) +
                         self.contrast_sim_kd(cross_sim_gt_TS, sim_gt_TT)) / 4
                # loss += (self.contrast_sim_kd(sim_other_SS, cross_sim_other_ST) +
                #          self.contrast_sim_kd(sim_other_TT, cross_sim_other_TS) +
                #          cross_kl_anchor) / 3
            else:
                # cross_sim_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), preds_T)
                # sim_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), preds_S)

                cross_sim_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), preds_S)
                sim_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), preds_T)

                # loss += (self.contrast_sim_kd(sim_SS, cross_sim_ST) +
                #          self.contrast_sim_kd(sim_TT, cross_sim_TS)) / 2
                loss += self.contrast_sim_kd(sim_TT, cross_sim_TS)

        return loss

    def contrast_sim_kd(self, s_logits, t_logits, dim=1, reduction='batchmean'):
        p_s = F.log_softmax(s_logits / self.tau, dim=dim)
        p_t = F.softmax(t_logits / self.tau, dim=dim)
        sim_dis = F.kl_div(p_s, p_t, reduction=reduction) * self.tau ** 2
        return sim_dis

class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        # self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.SyncBatchNorm(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


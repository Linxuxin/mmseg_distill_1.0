import torch
from torch import nn
import torch.nn.functional as F
import math
from ..builder import DISTILL_LOSSES

eps = 1e-7


@DISTILL_LOSSES.register_module()
class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, name, student_channels, teacher_channels, loss_weight, feat_dim=128, tau=0.07):
        super(CRDLoss, self).__init__()
        self.name = name
        self.loss_weight = loss_weight
        self.embed_s = Embed(student_channels, feat_dim)
        self.embed_t = Embed(teacher_channels, feat_dim)
        self.tau = tau
        self.num_classes = 19

    def forward(self, f_s, f_t, gt):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)

        # if self.align is not None:
        #     f_s = self.align(f_s)

        # f_s = F.normalize(f_s, p=2, dim=1)
        # f_t = F.normalize(f_t, p=2, dim=1)

        N, C, H, W = f_s.shape
        label = gt.unique()
        gt = F.interpolate(gt.float(), size=[H, W], mode='nearest').int()

        loss = 0
        for i in range(self.num_classes):
            if i in label:
                mask_gt = (gt == i).float()
                neg_mask = 1 - mask_gt

                # sample pos
                sampled_pos = []
                neg_idx = []
                ptr = 0
                for b in range(N):
                    num_pos_in_batch = len(mask_gt.view(N, -1)[b].nonzero())
                    if num_pos_in_batch == 0:
                        continue
                    sampled_idx = torch.randint(0, num_pos_in_batch, (1, 1))
                    sampled_pos.append(sampled_idx[0][0])

                    nz_neg = neg_mask.view(N, -1).nonzero()
                    num_neg_in_batch = len(neg_mask.view(N, -1)[b].nonzero())
                    neg_idx.append(nz_neg[ptr:ptr+num_neg_in_batch, 1])
                    ptr += num_neg_in_batch

                out_s, out_t = self.contrast(f_s, f_t, sampled_pos, neg_idx)
                s_loss = self.criterion(out_s)
                t_loss = self.criterion(out_t)
                loss += s_loss + t_loss

        return self.loss_weight * loss / len(label)-1

    def contrast(self, v1, v2, y, idx):
        N, C, H, W = v1.shape

        out_v1 = []
        out_v2 = []
        for b in range(len(y)):
            v1_pos_neg_idx = torch.zeros(len(idx[b]) + 1, dtype=torch.long)
            v2_pos_neg_idx = torch.zeros(len(idx[b]) + 1, dtype=torch.long)

            v1_pos_neg_idx[0] = y[b].data
            v1_pos_neg_idx[1:] = idx[b].data
            v2_pos_neg_idx[0] = y[b].data
            v2_pos_neg_idx[1:] = idx[b].data

            v1_flatten = v1.select(0, b).view(C, -1)
            v2_flatten = v2.select(0, b).view(C, -1)

            # sample
            weight_v1 = torch.index_select(v1_flatten, 1, v1_pos_neg_idx.cuda()).detach()
            weight_v1 = weight_v1.permute(1, 0)
            h_v2 = torch.matmul(weight_v1, v2_flatten)
            h_v2 = torch.exp(torch.div(h_v2, self.tau))

            Z_v2 = (h_v2.mean() * H * W).clone().detach().item()
            h_v2 = torch.div(h_v2, Z_v2).contiguous()
            out_v2.append(h_v2)

            # sample
            weight_v2 = torch.index_select(v2_flatten, 1, v2_pos_neg_idx.cuda()).detach()
            weight_v2 = weight_v2.permute(1, 0)
            h_v1 = torch.matmul(weight_v2, v1_flatten)
            h_v1 = torch.exp(torch.div(h_v1, self.tau))

            Z_v1 = (h_v1.mean() * H * W).clone().detach().item()
            h_v1 = torch.div(h_v1, Z_v1).contiguous()
            out_v1.append(h_v1)

        # out_v1 = torch.cat(out_v1, dim=0)
        # out_v2 = torch.cat(out_v2, dim=0)

        return out_v1, out_v2

    def criterion(self, x):
        loss = 0
        N = len(x)
        for b in range(N):
            x0 = x[b]
            m = x0.size(0) - 1

            # noise distribution
            Pn = 1 / float(x0.size(1)*2975)

            # loss for positive pair
            P_pos = x0.select(0, 0)
            log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

            # loss for K negative pair
            P_neg = x0.narrow(0, 1, m)
            log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

            loss += - (log_D1.mean(0) + log_D0.sum(0).mean())

        if N > 0:
            return loss / N
        else:
            return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """

    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        # self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.SyncBatchNorm(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


# class ContrastMemory(nn.Module):
#     """
#     memory buffer that supplies large amount of negative samples.
#     """
#
#     def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
#         super(ContrastMemory, self).__init__()
#         self.multinomial.cuda()
#         self.K = K
#
#     def forward(self, v1, v2, y, idx=None):
#         N, C, H, W = v1.shape
#
#         # sample
#         weight_v1 = torch.index_select(v1.view(N, C, -1), 0, idx[0].cuda).detach()
#         weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
#         out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
#         out_v2 = torch.exp(torch.div(out_v2, T))
#         # sample
#         weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
#         weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
#         out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
#         out_v1 = torch.exp(torch.div(out_v1, T))
#
#         # set Z if haven't been set yet
#         if Z_v1 < 0:
#             self.params[2] = out_v1.mean() * outputSize
#             Z_v1 = self.params[2].clone().detach().item()
#             print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
#         if Z_v2 < 0:
#             self.params[3] = out_v2.mean() * outputSize
#             Z_v2 = self.params[3].clone().detach().item()
#             print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))
#
#         # compute out_v1, out_v2
#         out_v1 = torch.div(out_v1, Z_v1).contiguous()
#         out_v2 = torch.div(out_v2, Z_v2).contiguous()
#
#         # update memory
#         with torch.no_grad():
#             l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
#             l_pos.mul_(momentum)
#             l_pos.add_(torch.mul(v1, 1 - momentum))
#             l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
#             updated_v1 = l_pos.div(l_norm)
#             self.memory_v1.index_copy_(0, y, updated_v1)
#
#             ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
#             ab_pos.mul_(momentum)
#             ab_pos.add_(torch.mul(v2, 1 - momentum))
#             ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
#             updated_v2 = ab_pos.div(ab_norm)
#             self.memory_v2.index_copy_(0, y, updated_v2)
#
#         return out_v1, out_v2


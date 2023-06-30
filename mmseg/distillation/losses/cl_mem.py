import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class ContrastMemLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, name, student_channels=19, teacher_channels=19, loss_weight=0.5, ignore_label=255, max_views=100,
                 max_samples=1024, temperature=0.07, base_temperature=0.07, feat_dim=256, r=5000,):
        super(ContrastMemLoss, self).__init__()
        self.name = name
        self.loss_weight = loss_weight
        self.num_classes = 19
        self.ignore_label = ignore_label
        self.max_views = max_views
        self.max_samples = max_samples
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.r = r
        self.pixel_update_freq = 10

        self.embed_s = Embed(student_channels, feat_dim)
        self.embed_t = Embed(teacher_channels, feat_dim)

        # self.register_buffer("teacher_segment_queue", torch.randn(self.num_classes, self.r, feat_dim))
        # self.teacher_segment_queue = nn.functional.normalize(self.teacher_segment_queue, p=2, dim=2)
        # self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        # self.register_buffer("teacher_pixel_queue", torch.randn(self.num_classes, self.r, feat_dim))
        # self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=2)
        # self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

    def forward(self, preds_S, preds_T, logits_S, gt):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        preds_S = self.embed_s(preds_S)
        preds_T = self.embed_t(preds_T)

        N, C, H, W = preds_S.shape
        # gt = F.interpolate(gt.float(), size=[H, W], mode='bilinear', align_corners=False).int()
        _, predict = torch.max(logits_S, 1)

        labels = gt.float().clone()
        labels = torch.nn.functional.interpolate(labels, (H, W), mode='nearest')
        labels = labels.long()
        assert labels.shape[-1] == preds_S.shape[-1], '{} {}'.format(labels.shape, preds_S.shape)

        ori_t_fea = preds_T
        ori_labels = labels

        labels = labels.contiguous().view(N, -1)
        predict = predict.contiguous().view(N, -1)
        feats_S = preds_S.permute(0, 2, 3, 1)
        feats_S = feats_S.contiguous().view(feats_S.shape[0], -1, feats_S.shape[-1])
        feats_T = preds_T.permute(0, 2, 3, 1)
        feats_T = feats_T.contiguous().view(feats_T.shape[0], -1, feats_T.shape[-1])

        feats_S_, feats_T_, labels_ = self._hard_anchor_sampling(feats_S, feats_T, labels, predict)

        loss_s = self._contrastive(feats_S_, feats_T_, labels_)
        loss_t = self._contrastive(feats_T_, feats_S_, labels_)
        loss = loss_s + loss_t

        # self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone())

        return self.loss_weight * loss

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

    def _contrastive(self, feats_S, feats_T, labels_):
        anchor_num, n_view = feats_S.shape[0], feats_S.shape[1]

        # X_contrast, y_contrast = self._sample_negative(feats_T)
        # y_contrast = y_contrast.contiguous().view(-1, 1).detach()
        # contrast_count = 1
        # contrast_feature = X_contrast.detach()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_T, dim=1), dim=0).detach()

        anchor_feature = torch.cat(torch.unbind(feats_S, dim=1), dim=0)
        anchor_count = n_view

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-6)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _hard_anchor_sampling(self, X_S, X_T, y_hat, y):
        batch_size, feat_dim = X_S.shape[0], X_S.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_S_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_T_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_S_[X_ptr, :, :] = X_S[ii, indices, :].squeeze(1)
                X_T_[X_ptr, :, :] = X_T[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_S_, X_T_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _dequeue_and_enqueue(self, keys, labels):
        segment_queue = self.teacher_segment_queue
        pixel_queue = self.teacher_pixel_queue

        # keys = self.concat_all_gather(keys)
        # labels = self.concat_all_gather(labels)

        batch_size, feat_dim, H, W = keys.size()

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.r

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.r:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + K) % self.r


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        # self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)
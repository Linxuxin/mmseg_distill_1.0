import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class CIRKDMemLoss(nn.Module):

    def __init__(self, student_channels, teacher_channels, name, tau=0.1, pooling=False, loss_weight=1.0,
                 region_memory_size=2000, pixel_memory_size=20000, pixel_contrast_size=4096, region_contrast_size=1024,
                 contrast_kd_temperature=1.0, contrast_temperature=0.1, ignore_label=255, loss_weight_pixel=0.1,
                 loss_weight_region=0.1, feat_dim=256, max_views=100, max_samples=1024, anchor_select_mode='semi',
                 anchor_sample_mode='random', pixel_positive_size=1024, pixel_negative_size=2048,
                 neg_sample_mode='semi', num_classes=19):
        super(CIRKDMemLoss, self).__init__()
        self.name = name
        self.tau = tau
        self.pooling = pooling
        self.loss_weight = loss_weight
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.dim = teacher_channels
        self.ignore_label = ignore_label
        self.max_views = max_views
        self.max_samples = max_samples
        self.loss_weight_pixel = loss_weight_pixel
        self.loss_weight_region = loss_weight_region
        self.anchor_select_mode = anchor_select_mode
        self.anchor_sample_mode = anchor_sample_mode
        self.neg_sample_mode = neg_sample_mode

        self.project_head = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.SyncBatchNorm(teacher_channels),
            nn.ReLU(True),
            nn.Conv2d(teacher_channels, teacher_channels, 1, bias=False)
        )

        self.num_classes = num_classes
        self.region_memory_size = region_memory_size
        self.pixel_memory_size = pixel_memory_size
        self.pixel_update_freq = 10
        self.pixel_contrast_size = pixel_contrast_size // self.num_classes + 1
        self.pixel_negative_size = pixel_negative_size // self.num_classes + 1
        self.pixel_positive_size = pixel_positive_size // self.num_classes + 1
        self.region_contrast_size = region_contrast_size // self.num_classes + 1

        # self.embed_s = Embed(student_channels, feat_dim)
        # self.embed_t = Embed(teacher_channels, feat_dim)

        self.register_buffer("teacher_segment_queue", torch.randn(self.num_classes, self.region_memory_size, self.dim))
        self.teacher_segment_queue = nn.functional.normalize(self.teacher_segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("teacher_pixel_queue", torch.randn(self.num_classes, self.pixel_memory_size, self.dim))
        self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=2)
        self.register_buffer("t_pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        # self.register_buffer("student_pixel_queue", torch.randn(self.num_classes, self.pixel_memory_size, feat_dim))
        # self.student_pixel_queue = nn.functional.normalize(self.student_pixel_queue, p=2, dim=2)
        # self.register_buffer("s_pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

        # self.register_buffer("X_pixel_contrast", torch.zeros(self.max_samples, self.num_classes * (
        #             self.pixel_negative_size + self.pixel_positive_size), feat_dim).float())
        # self.register_buffer("y_pixel_contrast", torch.zeros(self.max_samples, self.num_classes * (
        #             self.pixel_negative_size + self.pixel_positive_size), 1).float())

    def _sample_negative(self, Q, index, X_=None, y_=None, mode='random'):
        class_num, cache_size, feat_size = Q.shape

        if mode == 'random':
            contrast_size = index.size(0)
            X_ = torch.zeros((class_num * contrast_size, feat_size)).float().cuda()
            y_ = torch.zeros((class_num * contrast_size, 1)).float().cuda()
            sample_ptr = 0

            for ii in range(class_num):
                this_q = Q[ii, index, :]
                X_[sample_ptr:sample_ptr + contrast_size, ...] = this_q
                y_[sample_ptr:sample_ptr + contrast_size, ...] = ii
                sample_ptr += contrast_size

            return X_, y_
        elif mode == 'semi':
            anchor_size = index.size(0)
            contrast_size = index.size(2)
            sample_ptr = 0
            for ii in range(class_num):
                this_q = Q[ii][index[:, ii, :]]
                X_[:anchor_size, sample_ptr:sample_ptr + contrast_size, ...] = this_q
                y_[:anchor_size, sample_ptr:sample_ptr + contrast_size, ...] = ii
                sample_ptr += contrast_size

            return X_[:anchor_size], y_[:anchor_size]
        else:
            raise Exception

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def _dequeue_and_enqueue(self, keys, labels, segment_queue, pixel_queue, pixel_queue_ptr):
        keys = self.concat_all_gather(keys)
        labels = self.concat_all_gather(labels)

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
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.region_memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= self.pixel_memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + K) % self.pixel_memory_size

    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits / self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits / self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature ** 2
        return sim_dis

    def forward(self, s_feats, t_feats, logits_S=None, logits_T=None, labels=None):
        N, C, H, W = s_feats.shape

        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        # s_feats = self.embed_s(s_feats)
        # t_feats = self.embed_t(t_feats)

        _, predict_S = torch.max(logits_S, 1)
        # _, predict_T = torch.max(logits_T, 1)

        # confidence_map_S = torch.max(F.softmax(logits_S, dim=1), dim=1)[0]
        # confidence_map_T = torch.max(F.softmax(logits_T, dim=1), dim=1)[0]

        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (H, W), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == s_feats.shape[-1], '{} {}'.format(labels.shape, s_feats.shape)

        ori_s_fea = s_feats
        ori_t_fea = t_feats
        ori_labels = labels

        batch_size = s_feats.shape[0]

        labels = labels.contiguous().view(N, -1)
        predict_S = predict_S.contiguous().view(batch_size, -1)
        # predict_T = predict_T.contiguous().view(batch_size, -1)

        idxs = (labels != self.ignore_label)
        s_feats = s_feats.permute(0, 2, 3, 1)
        s_feats = s_feats.contiguous().view(N, -1, s_feats.shape[-1])
        s_feats = s_feats[idxs, :]

        t_feats = t_feats.permute(0, 2, 3, 1)
        t_feats = t_feats.contiguous().view(N, -1, t_feats.shape[-1])
        t_feats = t_feats[idxs, :]

        self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone(), self.teacher_segment_queue,
                                  self.teacher_pixel_queue, self.t_pixel_queue_ptr)

        # self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone(), self.teacher_segment_queue,
        #                           self.teacher_pixel_queue, self.t_pixel_queue_ptr)
        # self._dequeue_and_enqueue(ori_s_fea.detach().clone(), ori_labels.detach().clone(), self.teacher_segment_queue,
        #                           self.student_pixel_queue, self.s_pixel_queue_ptr)

        if idxs.sum() == 0:  # just a trick to skip all ignored anchor embeddings
            return 0. * (s_feats ** 2).mean(), 0. * (s_feats ** 2).mean()

        # feats_S_, labels_, s_avg_hard, s_avg_easy = self._hard_anchor_sampling(s_feats, labels, predict_S,
        #                                                                        confidence_map_S)
        # feats_T_, labels_, t_avg_hard, t_avg_easy = self._hard_anchor_sampling(t_feats, labels, predict_T,
        #                                                                        confidence_map_T)

        # feats_S_, feats_T_, labels_, avg_hard, avg_easy = self._hard_anchor_sampling(s_feats, t_feats, labels,
        #                                                                              predict_S, predict_T,
        #                                                                              confidence_map_S, confidence_map_T)

        # feats_S_, feats_T_, labels_ = self._segment_anchor(s_feats, t_feats, labels)

        # if self.neg_sample_mode == 'random':
        #     # random
        #     class_num, pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        #     perm = torch.randperm(pixel_queue_size)
        #     pixel_index = perm[:self.pixel_contrast_size]
        #     t_X_pixel_contrast, t_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index, mode='random')
        #     s_X_pixel_contrast, s_y_pixel_contrast = self._sample_negative(self.student_pixel_queue, pixel_index, mode='random')
        # elif 'semi' in self.neg_sample_mode:
        #     pixel_index_T = self._neg_idx_sampling(feats_S_, self.teacher_pixel_queue, mode=self.neg_sample_mode)
        #     t_X_pixel_contrast, t_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index_T,
        #                                                                    self.X_pixel_contrast, self.y_pixel_contrast,
        #                                                                    mode='semi')
        #     pixel_index_S = self._neg_idx_sampling(feats_T_, self.student_pixel_queue, mode=self.neg_sample_mode)
        #     s_X_pixel_contrast, s_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index_S,
        #                                                                    self.X_pixel_contrast, self.y_pixel_contrast,
        #                                                                    mode='semi')
        # else:
        #     raise Exception
        #
        # loss_s = self._contrastive(feats_S_, t_X_pixel_contrast, labels_, t_y_pixel_contrast, mode=self.neg_sample_mode)
        # loss_t = self._contrastive(feats_T_, s_X_pixel_contrast, labels_, s_y_pixel_contrast, mode=self.neg_sample_mode)

        class_num, pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        perm = torch.randperm(pixel_queue_size)
        pixel_index = perm[:self.pixel_contrast_size]
        t_X_pixel_contrast, t_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index)

        t_pixel_logits = torch.div(torch.mm(t_feats, t_X_pixel_contrast.T), self.contrast_temperature)
        s_pixel_logits = torch.div(torch.mm(s_feats, t_X_pixel_contrast.T), self.contrast_temperature)

        class_num, region_queue_size, feat_size = self.teacher_segment_queue.shape
        perm = torch.randperm(region_queue_size)
        region_index = perm[:self.region_contrast_size]
        t_X_region_contrast, _ = self._sample_negative(self.teacher_segment_queue, region_index)

        t_region_logits = torch.div(torch.mm(t_feats, t_X_region_contrast.T), self.contrast_temperature)
        s_region_logits = torch.div(torch.mm(s_feats, t_X_region_contrast.T), self.contrast_temperature)

        pixel_sim_dis = self.contrast_sim_kd(s_pixel_logits, t_pixel_logits.detach())
        region_sim_dis = self.contrast_sim_kd(s_region_logits, t_region_logits.detach())

        return self.loss_weight_pixel * pixel_sim_dis + self.loss_weight_region * region_sim_dis

        # loss = self.loss_weight * (loss_t + loss_s)
        # # num_samples = {"s_num_hard": s_avg_hard, "s_num_easy": s_avg_easy, "t_num_hard": t_avg_hard,
        # #                "t_num_easy": t_avg_easy}
        # num_samples = {"num_hard": avg_hard, "num_easy": avg_easy}
        return loss

    def _contrastive(self, feats_S, feats_T, labels_, y_contrast, mode='random'):
        anchor_num, n_view = feats_S.shape[0], feats_S.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)

        contrast_count = 1
        contrast_feature = feats_T.detach()

        anchor_feature = torch.cat(torch.unbind(feats_S, dim=1), dim=0)
        anchor_count = n_view

        if 'semi' in mode:
            mask = torch.eq(labels_.repeat(n_view, 1).unsqueeze(1), torch.transpose(y_contrast, 1, 2)).squeeze(1).float().cuda()
            anchor_dot_contrast = torch.bmm(anchor_feature.unsqueeze(1), feats_T.transpose(1, 2)).squeeze(1)
        elif mode == 'random':
            mask = torch.eq(labels_, torch.transpose(y_contrast, 0, 1)).float().cuda()
            mask = mask.repeat(anchor_count, contrast_count)

            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                            self.contrast_temperature)
        else:
            raise Exception

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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

        loss = -mean_log_prob_pos.mean()

        return loss

    def _hard_anchor_sampling(self, X_S, X_T, y_hat, y_S, y_T, c_map_S, c_map_T):
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
        list_hard = []
        list_easy = []
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y_S = y_S[ii]
            this_y_T = y_T[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                s_hard_anchor = ((this_y_hat == cls_id) & (this_y_S != this_y_T))
                s_easy_anchor = ((this_y_hat == cls_id) & (this_y_S == this_y_T))
                # t_hard_anchor = ((this_y_hat == cls_id) & (this_y_T != cls_id))
                # t_easy_anchor = ((this_y_hat == cls_id) & (this_y_T == cls_id))
                # hard_anchor = s_hard_anchor if t_easy_anchor.sum() == 0 or self.anchor_sample_mode == 'random' else s_hard_anchor * t_easy_anchor
                hard_indices = s_hard_anchor.nonzero()
                easy_indices = s_easy_anchor.nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                list_hard.append(num_hard)
                list_easy.append(num_easy)

                if self.anchor_select_mode == 'semi':
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
                elif self.anchor_select_mode == 'hard':
                    if num_hard >= n_view:
                        num_hard_keep = n_view
                        num_easy_keep = 0
                    else:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                elif self.anchor_select_mode == 'easy':
                    if num_hard >= n_view:
                        num_easy_keep = n_view
                        num_hard_keep = 0
                    else:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                else:
                    raise Exception

                if self.anchor_sample_mode == 'random':
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)
                elif self.anchor_sample_mode == 'prob':
                    # TODO
                    h_prob = s_hard_anchor * (1 - c_map_S[ii].view(-1))
                    e_prob = s_easy_anchor * (1 - c_map_S[ii].view(-1))
                    if num_easy_keep > 0 and num_hard_keep == 0:
                        easy_indices = torch.multinomial(e_prob, num_easy_keep)
                        indices = torch.cat((hard_indices, easy_indices.unsqueeze(1)), dim=0)
                    elif num_hard_keep > 0 and num_easy_keep == 0:
                        hard_indices = torch.multinomial(h_prob, num_hard_keep)
                        indices = torch.cat((hard_indices.unsqueeze(1), easy_indices), dim=0)
                    elif num_easy_keep > 0 and num_hard_keep > 0:
                        easy_indices = torch.multinomial(e_prob, num_easy_keep)
                        hard_indices = torch.multinomial(h_prob, num_hard_keep)
                        indices = torch.cat((hard_indices, easy_indices), dim=0).unsqueeze(1)
                else:
                    raise Exception

                X_S_[X_ptr, :, :] = X_S[ii, indices, :].squeeze(1)
                X_T_[X_ptr, :, :] = X_T[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        avg_hard = sum(list_hard)/len(list_hard)
        avg_easy = sum(list_easy)/len(list_easy)

        return X_S_, X_T_, y_, avg_hard, avg_easy
        # return X_S_, y_, avg_hard, avg_easy

    def _neg_idx_sampling(self, anchor, queue, num_topk=0.1, mode='semi_random'):
        class_num, pixel_queue_size, feat_size = queue.shape
        sim_pn = torch.mm(anchor.view(anchor.shape[0] * anchor.shape[1], feat_size),
                          queue.view(class_num * pixel_queue_size, feat_size).permute(1, 0))
        sim_pn = sim_pn.view(-1, class_num, pixel_queue_size)
        top_k = int(pixel_queue_size * num_topk)
        prob_neg, idx_neg = sim_pn.topk(top_k, dim=2)
        prob_pos, idx_pos = (sim_pn * (-1)).topk(top_k, dim=2)

        assert mode in ['semi_random', 'semi_prob']
        if 'random' in mode:
            perm_pos = torch.randperm(top_k)
            perm_neg = torch.randperm(top_k)
            pixel_index_pos = perm_pos[:self.pixel_positive_size]
            pixel_index_neg = perm_neg[:self.pixel_negative_size]
            pixel_index = torch.cat([idx_pos[:, :, pixel_index_pos], idx_neg[:, :, pixel_index_neg]], dim=2)
        elif 'prob' in mode:
            # list_pixel_index = []
            # for ii in range(sim_pn.shape[0]):
            #     pixel_index_neg = torch.multinomial(prob_neg[ii], self.pixel_negative_size)
            #     pixel_index_pos = torch.multinomial(prob_pos[ii], self.pixel_positive_size)
            #     pixel_index = torch.cat([pixel_index_pos, pixel_index_neg], dim=1)
            #     list_pixel_index.append(pixel_index)
            #
            # pixel_index = torch.stack(list_pixel_index, dim=0)

            pixel_index_neg = torch.multinomial(prob_neg.view(-1, class_num*top_k), self.pixel_negative_size)
            pixel_index_pos = torch.multinomial(prob_pos, self.pixel_positive_size)
            pixel_index = torch.cat([pixel_index_pos, pixel_index_neg], dim=1)
        else:
            raise Exception

        return pixel_index

    # def _segment_anchor(self, X_S, X_T, y_hat):


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


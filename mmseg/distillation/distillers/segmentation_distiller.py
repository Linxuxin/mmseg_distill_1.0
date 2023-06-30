import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models import build_segmentor
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from mmseg.core import add_prefix
from mmseg.models.backbones.sagan import Discriminator
from mmseg.distillation.losses.distill_discriminator_loss import CriterionAdditionalGP, CriterionAdv


@DISTILLER.register_module()
class SegmentationDistiller(BaseSegmentor):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 use_logit=True,
                 use_adv=False,
                 num_classes=19,):

        super(SegmentationDistiller, self).__init__()

        self.use_adv = use_adv
        self.num_classes = num_classes
        if self.use_adv:
            self.D_model = Discriminator(preprocess_GAN_mode=1, input_channel=self.num_classes,
                                         batch_size=16, image_size=65, conv_dim=64)
            # load_D_model(D_resume=True, model=D_model, D_ckpt_path='./pretrained/Distriminator')
            # self.D_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, D_model.parameters()), lr=0.0004,
            #                                  betas=(0.9, 0.99))
            # self.D_model = D_model

            self.criterion_adv = CriterionAdv('hinge', loss_weight=0.1).cuda()
            self.criterion_AdditionalGP = CriterionAdditionalGP(self.D_model, lambda_gp=10.0).cuda()

        self.teacher = build_segmentor(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.use_logit = use_logit
        self.student= build_segmentor(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights()
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg      
        for item_loc in distill_cfg:
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])

    def D_model_parameters(self):
        return nn.ModuleList([self.D_model])

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self.student,
                       'auxiliary_head') and self.student.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self.student, 'decode_head') and self.student.decode_head is not None

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, img, img_metas, gt_semantic_seg):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        with torch.no_grad():
            self.teacher.eval()
            fea_t = self.teacher.extract_feat(img)
            if self.use_logit:
                output_t = self.teacher._decode_head_forward_test(fea_t, img_metas)
                logit_t = output_t[0]
                feat_after_psp_t = output_t[1]

        student_feat = self.student.extract_feat(img)
        output_s = self.student._decode_head_forward_test(student_feat, img_metas)
        logit_s = output_s[0]
        feat_after_psp_s = output_s[1]

        losses = self.student.decode_head.losses(logit_s, gt_semantic_seg)
        loss_decode = dict()
        loss_decode.update(add_prefix(losses, 'decode'))

        student_loss = dict()
        student_loss.update(loss_decode)

        if self.student.with_auxiliary_head:
            loss_aux = self.student._auxiliary_head_forward_train(
                student_feat, img_metas, gt_semantic_seg)
            student_loss.update(loss_aux)

        adv_loss = 0
        if self.use_adv:
            student_loss['loss_adv_g'] = self.distill_losses['loss_adv_g'](self.D_model(logit_s))

            adv_loss = self.criterion_adv(self.D_model(logit_s.detach()), self.D_model(logit_t.detach()))
            adv_loss += 0.1 * self.criterion_AdditionalGP(logit_s, logit_t)

        loss_name1 = 'loss_mgd_fea1'
        loss_name2 = 'loss_mgd_fea2'
        loss_name3 = 'loss_mgd_fea3'
        loss_name4 = 'loss_mgd_fea4'
        # if self.iter < 20000:
        student_loss[loss_name1] = self.distill_losses[loss_name1](student_feat[-1], fea_t[-1].detach())

        # student_loss[loss_name2] = self.distill_losses[loss_name2](student_feat[-2], fea_t[-2].detach())
        # student_loss[loss_name3] = self.distill_losses[loss_name3](student_feat[-3], fea_t[-3].detach())
        # student_loss[loss_name4] = self.distill_losses[loss_name4](student_feat[-4], fea_t[-4].detach())

        student_loss['loss_idd_f3'] = self.distill_losses['loss_idd_f3'](student_feat[-2], fea_t[-2].detach(),
                                                                   gt_semantic_seg)
        # student_loss['loss_idd_f4'] = self.distill_losses['loss_idd_f4'](student_feat[-1], fea_t[-1].detach(),
        #                                                            gt_semantic_seg)
        # student_loss['loss_idd_ap'] = self.distill_losses['loss_idd_ap'](feat_after_psp_s, feat_after_psp_t.detach(),
        #                                                                  gt_semantic_seg)

        # student_loss['loss_ifvd'] = self.distill_losses['loss_ifvd'](feat_after_psp_s, feat_after_psp_t.detach(), gt_semantic_seg)

        # student_loss['loss_cl_mem'] = self.distill_losses['loss_cl_mem'](student_feat[-1], fea_t[-1], logit_s, gt_semantic_seg)

        # student_loss['loss_cirkd'] = self.distill_losses['loss_cirkd'](feat_after_psp_s, feat_after_psp_t.detach())
        # student_loss['loss_cirkd_mem'] = self.distill_losses['loss_cirkd_mem'](feat_after_psp_s, feat_after_psp_t.detach(),
        #                                                                        logit_s, logit_t, gt_semantic_seg)

        # student_loss['loss_pa'] = self.distill_losses['loss_pa'](feat_after_psp_s, feat_after_psp_t.detach())

        # student_loss['loss_svd_features_features'] = self.distill_losses['loss_svd_features'](student_feat[-1], fea_t[-1].detach())

        # student_loss['loss_dkd_features'] = self.distill_losses['loss_dkd_features'](student_feat[-1],
        #                                                                              fea_t[-1].detach(),
        #                                                                              gt_semantic_seg)
        #
        # student_loss['loss_cl'] = self.distill_losses['loss_cl'](student_feat[-1], logit_s, gt_semantic_seg)
        
        if self.use_logit:
            # student_loss['loss_ifvd'] = self.distill_losses['loss_ifvd'](logit_s, logit_t, gt_semantic_seg)
            # student_loss['loss_idd'] = self.distill_losses['loss_idd'](logit_s, logit_t.detach(), gt_semantic_seg)
            # student_loss['loss_crd'] = self.distill_losses['loss_crd'](logit_s, logit_t, gt_semantic_seg)
            # if self.iter >= 20000:
            # student_loss['loss_cl'] = self.distill_losses['loss_cl'](student_feat[-1], fea_t[-1], logit_s, gt_semantic_seg)

            # if self.iter < 32000:

            num_samples = dict()
            # student_loss['loss_cirkd_cl_mem'], num_samples = self.distill_losses['loss_cirkd_cl_mem'](
            #     student_feat[-1], fea_t[-1],
            #     logit_s, logit_t, gt_semantic_seg)

            # student_loss['loss_dkd_pixel'] = self.distill_losses['loss_dkd_pixel'](logit_s, logit_t.detach(), gt_semantic_seg)
            # student_loss['loss_dkd_channel'] = self.distill_losses['loss_dkd_channel'](logit_s, logit_t.detach(), gt_semantic_seg)
            
            # student_loss['loss_dist'] = self.distill_losses['loss_dist'](logit_s, logit_t.detach())

            # student_loss['loss_pi'] = self.distill_losses['loss_pi'](logit_s, logit_t.detach())

            # N, C, H, W = logit_s.shape
            # softmax_pred_T = F.softmax(logit_t.view(-1, W * H) / 4, dim=1)
            # logsoftmax = torch.nn.LogSoftmax(dim=1)
            # loss = torch.sum(softmax_pred_T *
            #                 logsoftmax(logit_t.view(-1, W * H) / 4) -
            #                 softmax_pred_T *
            #                 logsoftmax(logit_s.view(-1, W * H) / 4)) * (
            #                     4**2)
            #
            # student_loss['loss_logit'] = 3 * loss / (C * N)

        # return student_loss
        return student_loss, adv_loss, num_samples
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
    def encode_decode(self, img, img_metas):
        return self.student.encode_decode(img, img_metas)

def load_D_model(D_resume, model, D_ckpt_path):
    if D_resume:
        if os.path.isfile(D_ckpt_path):
            checkpoint = torch.load(D_ckpt_path)
            last_step = checkpoint['step'] if 'step' in checkpoint else None
            model.load_state_dict(checkpoint['state_dict'])


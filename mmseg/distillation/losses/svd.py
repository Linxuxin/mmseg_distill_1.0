import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class SVDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 loss_weight=1.0,
                 num_decomposed=1000,
                 ):
        super(SVDLoss, self).__init__()
        self.name = name
        self.loss_weight = loss_weight
        self.num_decomposed = num_decomposed

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_T.shape

        # ss, su, sv = torch.linalg.svd(preds_S.view(-1, H*W), full_matrices=False)
        # ts, tu, tv = torch.linalg.svd(preds_T.view(-1, H*W), full_matrices=False)
        ss, su, sv = torch.svd_lowrank(preds_S.view(-1, H * W), q=self.num_decomposed)
        ts, tu, tv = torch.svd_lowrank(preds_T.view(-1, H * W), q=self.num_decomposed)

        # ss, su, sv = ss[:, :self.num_decomposed], su[:self.num_decomposed], sv[:, :self.num_decomposed]
        # ts, tu, tv = ts[:, :self.num_decomposed], tu[:self.num_decomposed], tv[:, :self.num_decomposed]
        # sv = sv[:, :self.num_decomposed]
        # tv = tv[:, :self.num_decomposed]

        loss_mse = nn.MSELoss(reduction='sum')

        loss = loss_mse(sv, tv) / N

        return self.loss_weight * loss

import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class DISTLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, name, beta=1.0, gamma=1.0):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.name = name

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        assert preds_S.ndim in (2, 4)
        if preds_S.ndim == 4:
            num_classes = preds_S.shape[1]
            preds_S = preds_S.transpose(1, 3).reshape(-1, num_classes)
            preds_T = preds_T.transpose(1, 3).reshape(-1, num_classes)
        preds_S = preds_S.softmax(dim=1)
        preds_T = preds_T.softmax(dim=1)
        inter_loss = inter_class_relation(preds_S, preds_T)
        intra_loss = intra_class_relation(preds_S, preds_T)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
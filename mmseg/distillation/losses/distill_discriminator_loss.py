import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import scipy.ndimage as nd
from ..builder import DISTILL_LOSSES


class CriterionAdv(nn.Module):
    def __init__(self, adv_type, loss_weight, discriminator=True):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type
        self.loss_weight = loss_weight
        self.discriminator = discriminator

    def forward(self, d_out_S, d_out_T):
        assert d_out_S[0].shape == d_out_T[0].shape, 'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return self.loss_weight * (d_loss_real + d_loss_fake)


class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S.shape == d_in_T.shape,'the output dim of D with teacher and student as input differ'

        real_images = d_in_T
        fake_images = d_in_S
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out[0].size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss


@DISTILL_LOSSES.register_module()
class CriterionAdvForG(nn.Module):
    def __init__(self, name, adv_type, loss_weight, discriminator=True):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.name = name
        self.adv_loss = adv_type
        self.loss_weight = loss_weight
        self.discriminator = discriminator

    def forward(self, d_out_S):
        g_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return self.loss_weight * g_loss_fake
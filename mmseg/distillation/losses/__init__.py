from .mgd import  FeatureLoss
from .dkd import DecoupledKD
from .pi import PixelWiseLoss
from .distill_discriminator_loss import CriterionAdvForG, CriterionAdv
from .idd import InterClassLoss
from .svd import SVDLoss
from .ifvd import IFVDLoss
from .dist import DISTLoss
from .cl import ContrastLoss
from .crd import CRDLoss
from .cirkd import CIRKDLoss
from .cirkd_mem import CIRKDMemLoss
from .cl_mem import ContrastMemLoss
from .pa import PairWiseLoss
__all__ = [
    'FeatureLoss',
    'DecoupledKD',
    'PixelWiseLoss',
    'InterClassLoss',
    'SVDLoss',
    'IFVDLoss',
    'DISTLoss',
    'ContrastLoss',
    'CRDLoss',
    'CIRKDLoss',
    'CIRKDMemLoss',
    'ContrastMemLoss',
    'CriterionAdvForG',
    'PairWiseLoss'
]

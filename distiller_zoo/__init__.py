from .DKD import DKDloss
from .KD import DistillKL as DistillKL_Simple
from .PKT import PKT
from .SP import Similarity
from .UPKD import DistillKL, UPKDLoss
from .VID import VIDLoss

__all__ = [
    "DKDloss",
    "DistillKL",
    "DistillKL_Simple",
    "PKT",
    "Similarity",
    "UPKDLoss",
    "VIDLoss",
]

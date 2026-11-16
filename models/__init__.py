"""Model registry for the selected experiments.

The `model` and `args` fields in configs/ select the class and constructor.
Selected classes: PyramidDCN; DCNVariant; MLDCN/NarrowMLDCN;
CrossOnlyMaskNet/AlternativeMaskNet; FCNP; GDCN; QNN/PyramidQNN;
FinalMLP/PyramidFinalMLP; FiBiNetPP/PyramidFiBiNetPP.

The original modules also retain reusable components and earlier variants.
ECN is a branch of FCNP, not a separately reported baseline. See
`docs/TABLE2_MODELS.md` for the precise selected architectures.
"""
from .base import BaseModel
from .dcnv2 import DCNv2
from .ecn import ECN, PyramidECN
from .fcn import FCN, PyramidFCN
from .fibinetpp import FiBiNetPP, PyramidFiBiNetPP
from .finalmlp import FinalMLP, PyramidFinalMLP
from .gdcn import GDCN
from .masknet import MaskNet, PyramidMaskNet
from .mldcn import MLDCN, PyramidMLDCN
from .pyramiddcn import PyramidDCN
from .qnn import QNN, PyramidQNN

MODELS = {cls.__name__: cls for cls in (
    PyramidDCN, DCNv2, MLDCN, PyramidMLDCN, MaskNet, PyramidMaskNet, QNN, PyramidQNN, ECN, PyramidECN,
    FCN, PyramidFCN, GDCN, FinalMLP, PyramidFinalMLP, FiBiNetPP, PyramidFiBiNetPP)}

from .dcnv2 import DCNVariant
from .fcn import FCNP
from .masknet_serial import CrossOnlyMaskNet, AlternativeMaskNet
from .mldcn_narrow import NarrowMLDCN
MODELS.update({cls.__name__: cls for cls in (DCNVariant, FCNP, CrossOnlyMaskNet, AlternativeMaskNet, NarrowMLDCN)})

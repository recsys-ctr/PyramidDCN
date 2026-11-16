"""Only models selected for Tables 1–4 are registered here."""
from .base import BaseModel
from .dcnv2 import DCNVariant
from .fcn import FCNP
from .fibinetpp import FiBiNetPP, PyramidFiBiNetPP
from .finalmlp import FinalMLP, PyramidFinalMLP
from .gdcn import GDCN
from .masknet_serial import CrossOnlyMaskNet, AlternativeMaskNet
from .mldcn import MLDCN
from .mldcn_narrow import NarrowMLDCN
from .pyramiddcn import PyramidDCN
from .qnn import QNN, PyramidQNN

MODELS = {cls.__name__: cls for cls in (
    PyramidDCN, DCNVariant, MLDCN, NarrowMLDCN, CrossOnlyMaskNet,
    AlternativeMaskNet, FCNP, GDCN, QNN, PyramidQNN, FinalMLP,
    PyramidFinalMLP, FiBiNetPP, PyramidFiBiNetPP)}

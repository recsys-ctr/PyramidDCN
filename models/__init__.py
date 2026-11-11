"""PyramidDCN and the nine baselines of the paper, one file per model.

The "model" field of a run configuration (configs/) is a class name; its "args" are the keyword arguments of that
class.

    Paper        File            Classes
    PyramidDCN   pyramiddcn.py   PyramidDCN
    DCNv2        dcnv2.py        DCNv2
    ML-DCN       mldcn.py        MLDCN, PyramidMLDCN
    MaskNet      masknet.py      MaskNet, PyramidMaskNet
    QNN-alpha    qnn.py          QNN, PyramidQNN
    ECN          ecn.py          ECN, PyramidECN
    FCN          fcn.py          FCN, PyramidFCN
    GDCN         gdcn.py         GDCN
    FinalMLP     finalmlp.py     FinalMLP, PyramidFinalMLP
    FiBiNet++    fibinetpp.py    FiBiNetPP, PyramidFiBiNetPP

Pyramid<Model> is the pyramidal conversion of <Model> in Table 2.  DCNv2 and GDCN take per-layer widths, so their
conversions are the same classes with narrowing cross_widths.  The Table 3 variants and the Table 4 widths are
keyword arguments of PyramidDCN.  base.py and layers.py hold the shared pieces.

Interfaces: PyramidDCN, MLDCN and PyramidMLDCN map the projected input x [B, W] to logits [B].  The other models
subclass BaseModel (FuxiCTR conventions): forward(dense, cats) -> logits [B] and loss(cats, y); train.py replaces
their embedding table by the shared projected input.
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

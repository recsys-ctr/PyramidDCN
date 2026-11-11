# Third-party sources and attribution

The controlled baseline ports retain their source descriptions in the model files of `models/`. They are adaptations for the common-input experiment harness, rather than independently tuned reproductions of the original packages. The original vendor repositories are not copied into this release.

| Source | Version or inspected commit | Applicable information |
| --- | --- | --- |
| [FuxiCTR](https://github.com/reczoo/FuxiCTR) | 2.2.0, `42fa7a6037eea4d8cd50db9a420f170c37b57b5d` | Apache-2.0. Includes shared blocks and MaskNet and FinalMLP ports; the DCNv2 and GDCN cross layers follow its implementations. |
| [FCN](https://github.com/salmon1802/FCN) | KDD'26, `8eaa8645eed045cea9c1e41eba097627b614f054`; historical master sources are identified in the port headers | Used ECN/LCN/FCNv2 files carry Apache-2.0 notices (FuxiCTR Library or salmon@github), despite the repository-level license field being unset. |
| [QNN](https://github.com/salmon1802/QNN) | inspected main `08c5db81e76b1a377708ce6e6164f3d7a3a2486d` | Apache-2.0. This inspected revision is not asserted to be the exact historical revision used for the original port. |
| [GDCN](https://github.com/anonctr/GDCN) | `5165518f2bc312274c1d593e62c4ad16a6a7f989` | Reference implementation identified in the port. FuxiCTR also provides an Apache-2.0 GDCN implementation. |
| [FiBiNet / FiBiNet++](https://github.com/recommendation-algorithm/FiBiNet) | `af5558bfe3915c266f233ae0c83770712ea96cbd` | Reference TensorFlow implementation; this release uses a PyTorch hand-port. No upstream repository-wide license was found at this revision. No vendor source directory is included. |

The Apache-2.0 text is provided in `LICENSES/Apache-2.0.txt` for applicable third-party portions. Original per-file copyright and attribution statements remain applicable. This repository does not apply a new blanket license to all implementations or grant additional rights to third-party work. Dataset access and licenses are separate; no raw benchmark records or trained checkpoints are distributed.

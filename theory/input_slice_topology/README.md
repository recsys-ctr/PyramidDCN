# Input-slice topology

The proof concerns the default crossing widths `[W,W/2,W/4]` with a learned input projection. It establishes:

- Exact coordinate-permutation equivalence, including affine layer normalization. Numerical spacing alone does not change the hypothesis class for an unrestricted learned projection.
- A disjoint-slice quartic outside the coefficient closure of every nested-slice model at the same widths and matrix cost, with independently learned projections. This separation omits layer normalization.
- A reverse exact-representation example, without a reverse approximation-closure claim.

Read [PROOF.md](PROOF.md) for the self-contained constructions, proofs, and limitations. This result does not prove that disjoint slices or narrowing always improve benchmark accuracy. Historical allocation counter-controls are reported in [width_allocation_audit](../../results/reference/width_allocation_audit/README.md).

# Learned-projection width-allocation separation

This proof combines an analytic block construction with three exact,
computer-assisted base cases. The verifier reconstructs their integer-point
Jacobians and checks nonzero determinants modulo the prime `1000003`; these
are algebraic certificates, not training results or floating-point ranks.

## Model and dimension

Let `W=4k`, `k≥3`, and let the raw embedded input be `y∈R^D`, `D≥W`. Each
model learns its own dense projection `P∈R^(W×D)`. Partition `x=Py` into
coordinates `(u,v,t)` of dimensions `(2k,k,k)`. The second and third input
slices select `u` and `v`; their numerical positions are immaterial under a
coordinate permutation. The first slice contains all coordinates.

For three unnormalized crosses of widths `[4k,2k,k]`, absorb the output head
into `B=diag(w)A3`. Write `M=A1` and `A=A2`, with shapes `4k×4k`, `2k×4k`,
and `k×2k` for `M,A,B`. Define

\[
 H_1=x\odot Mx,\quad L_2=u\odot B^\top v,\quad
 H_2=u\odot AH_1,\quad K=x\odot A^\top L_2,
 \qquad F=L_2^\top AH_1=v^\top BH_2.
\]

Here `⊙` is elementwise multiplication; `F` is the homogeneous quartic logit.
Biases and additive input skips do not change this quartic. Let `Z_k(D)` be
the complex algebraic closure of its coefficient family. We prove

\[
 \boxed{\dim Z_k(D)=WD+26k^2-10k.}
\]

The parameter count is `WD+26k²`. Invertible diagonal matrices `Q,T1,T2`
act without changing `F` by

\[
 P'=QP,\quad M'=T_1Q^{-1}MQ^{-1},\quad
 A'=T_2Q_U^{-1}AT_1^{-1},\quad B'=Q_V^{-1}BT_2^{-1},
\]

where `Q_U,Q_V` restrict `Q` to the indicated slices. This `10k`-dimensional
action is free when `P,M,A` have nonzero rows: fixing these matrices
successively forces `Q,T1,T2` to be identities. This proves the upper bound.

## Certified bases and block lift

The frozen bases have identity projection `P=I` and all-one heads. They certify:

| Local size a | Joint `(P,M,A,B)` Jacobian rank | Rank of `(H1,L2)` | Rank of `(K,gradient F,H2)` |
|---:|---:|---:|---:|
| 3 | 348 | 18 | 30 |
| 4 | 632 | 24 | 40 |
| 5 | 1000 | 30 | 50 |

Each row holds at one explicit integer point. A nonzero evaluation-Jacobian
minor also certifies coefficient-Jacobian rank, because evaluation is a linear
map on coefficients. Its nonzero modular determinant implies a nonzero
integer determinant. The joint ranks are `42a²−10a`.

Every `k≥3` is a sum of sizes `3,4,5`. Use the corresponding certified bases
in block-diagonal `M,A,B`, with `P=I`. Their pure-block tangent ranks add.
For two blocks of sizes `a,b`, differentiate all off-diagonal parameters.
Subscripts name the block receiving the linear transformation:

| Parameter direction | Derivative | Degrees in the two variable blocks |
|---|---|---|
| `P_(a←b)` | `(gradient F_a) x_b` | `(3,1)` |
| `M_(a←b)` | `K_a x_b` | `(3,1)` |
| `B_(b←a)` | `H2_a v_b` | `(3,1)` |
| `A_(a←b)` | `L2_a H1_b` | `(2,2)` |

Products here denote all componentwise pairings. Cubic-bank independence
makes the `(3,1)` sector have rank `8a·4b+2a·b=34ab`; reversing blocks
adds another `34ab`. Quadratic-bank independence makes the two `(2,2)`
tensor subspaces disjoint, adding `2a·4b+4a·2b=16ab`. Thus every block pair
contributes `84ab`. Different pairs and pure-block tangents have different
monomial supports. The total rank is

\[
 \sum_i(42a_i^2-10a_i)+84\sum_{i<j}a_ia_j=42k^2-10k.
\]

For `D>W`, use `P=[I,0]`. Each additional projection column contributes the
`W` independent polynomials `y_j partial_l F`, where `partial_l` differentiates
in projected coordinate `l`. Cubic-bank independence and
different new-variable supports make these directions independent. Adding
`W(D−W)` reaches the stated upper bound.

## Separately learned flat competitors

The pyramidal cross/head budget is `B0=26k²+k` multiply–accumulates (MACs).
Consider any constant crossing width `c`, arbitrary fixed coordinate gathers
(including repetitions), and separately learned projection of the same
width `W`. A rank-`ρ` matrix of shape `m×n` has parameter dimension
`ρ(m+n−ρ)`, at most its chosen dense or two-dense-factor cost
`min(mn,ρ(m+n))`. Input and hidden diagonal scalings remove `W+3c`
dimensions on each positive-rank stratum. Consequently, under core budget
`B0`, the flat quartic dimension is at most

\[
 WD-W+\min\{B_0-3c,\;4kc+2c^2-2c\}.
\]

Nonzero rows make the action generically free; irreducible determinantal
closures ensure exceptional zero rows cannot enlarge this bound.
Zero-rank maps have zero quartic. If `c>7k/3`, the first branch is below
`26k²−6k`; otherwise the second is at most `182k²/9−14k/3`, also strictly
below it. Thus every flat closure is lower-dimensional than `Z_k(D)`.
Width, rank, and gather choices are finite under the budget, so their union
remains closed and lower-dimensional. Both sides additionally pay `WD` MACs
for their own dense projection; total matrix FLOPs are twice the MACs.

## Finite prediction consequence and scope

Choose a real pyramid quartic outside that closed flat union. The polynomial
maps have real coefficients, and a nonzero polynomial on their parameter
domain cannot vanish at every real point. The grid
`G_D={α/4: α∈N_0^D, sum(α)≤4}` uniquely determines degree-at-most-four
polynomials, where `N_0` denotes nonnegative integers: the multivariate binomial
basis gives a triangular evaluation
matrix with unit diagonal. Assign uniform grid inputs and Bernoulli
probabilities `sigmoid(F)=1/(1+exp(−F))`. Input skips can be canceled with biases `−1`,
so the pyramid attains Bayes LogLoss. A flat sequence approaching that risk
would converge to every finite teacher logit, hence to its coefficients by
interpolation, contradicting quartic closure exclusion. Its optimal
population LogLoss is therefore strictly greater.

This is an existential separation for three unnormalized crosses with equal
projected width, dense projection cost, and dense/two-factor state maps.
It gives no universal accuracy dominance, quantitative practical gain,
optimality of halving or uniform spacing, sparse-circuit bound, or guarantee
for layer normalization or training.

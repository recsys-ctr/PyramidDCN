# Exact width-allocation certificates

This standalone kit verifies small algebraic witnesses used by the
learned-projection width-allocation argument. It requires Python 3.10 or newer
and NumPy. It uses no network connection, dataset, model checkpoint, or GPU.
The accompanying [mathematical proof](PROOF.md) gives the all-size block lift,
the comparison with separately learned flat projections, and its precise scope.

From the repository root, run:

```sh
python3 theory/width_allocation/verify_certificates.py
```

The script resolves its imports and fixtures relative to its own location, so
it also works when invoked from another working directory. Each modular
elimination phase has a 30-second limit, adjustable with
`--max-phase-seconds`. Verification normally takes a few seconds.

The verifier reconstructs the complete evaluation Jacobians from explicit
integer parameters and input points. It compares every derivative row against
an independently derived closed-form expression, rebuilds the local polynomial
banks, checks all frozen array hashes, verifies gauge and Euler identities, and
computes all nine nonzero minor determinants. It does not accept a recorded
`PASS` result as evidence.

## Objects being certified

For input width `4a` and crossing widths `[4a,2a,a]`, let the second and final
input slices be `u` and `v`. Write `M=A1`, `A=A2`, and `B=diag(w)A3`. At each
frozen point the learned input projection is the identity and the head is all
ones. Define:

```text
H1 = x * (M @ x)          # 4a quadratic polynomials
L2 = u * (B.T @ v)        # 2a quadratic polynomials
K  = x * (A.T @ L2)       # 4a cubic polynomials
H2 = u * (A @ H1)         # 2a cubic polynomials
F  = L2.T @ A @ H1        # homogeneous quartic
```

Here `*` denotes elementwise multiplication. The quadratic bank is `(H1,L2)`;
the cubic bank is `(K,gradient_x(F),H2)`.

| a | Joint `(P,M,A,B)` Jacobian rank | Quadratic bank rank | Cubic bank rank |
|---:|---:|---:|---:|
| 3 | 348 | 18 | 30 |
| 4 | 632 | 24 | 40 |
| 5 | 1000 | 30 | 50 |

All computations use the prime field with modulus `1000003`. Every rank lower
bound has a saved square minor with a nonzero determinant. Evaluation is a
linear map on polynomial coefficient vectors, so these evaluation-Jacobian
minors also certify coefficient-Jacobian rank lower bounds. A nonzero minor
modulo the prime proves that the corresponding integer minor is nonzero in
characteristic zero. The independent diagonal gauge directions provide the
matching joint-Jacobian upper bounds at the frozen points.

These witnesses concern the unnormalized homogeneous quartic backbone. They
do not measure CTR accuracy, describe trained effective ranks, prove universal
optimality of halving, or establish claims about layer normalization. The
all-size block-lift argument is given in [PROOF.md](PROOF.md).

## Files and integrity

`finite_field_network.py` implements scalar forward evaluation and reverse-mode
parameter derivatives. `verify_certificates.py` adds an independent derivative
formula and exact modular elimination. `certificates/a3.json`, `a4.json`, and
`a5.json` freeze all integer parameters, evaluation inputs, minor indices,
determinants, array shapes, and array hashes. Large Jacobian arrays are rebuilt
during verification rather than shipped as binary fixtures.

`SHA256SUMS` records the source and fixture hashes. The verifier checks that
manifest before doing algebra. Array hashes use row-major, signed 64-bit,
little-endian integer bytes, independently of the host machine's endianness.

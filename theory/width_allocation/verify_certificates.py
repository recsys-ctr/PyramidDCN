"""Rebuild and verify exact learned-projection width-allocation witnesses.

Run with Python and NumPy. Fixtures and imports are resolved beside this file,
so the current working directory does not matter. No network access is used.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('finite_field_network', ROOT / 'finite_field_network.py')
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes(order='C')).hexdigest()


def check_distribution_hashes():
    count = 0
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), 'Manifest path escapes the kit'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
        count += 1
    return count


def determinant(matrix, prime, limit):
    start = time.monotonic()
    a = np.array(matrix, dtype=np.int64, copy=True) % prime
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    answer = 1
    for column in range(len(a)):
        if time.monotonic() - start > limit:
            raise TimeoutError('Modular determinant phase exceeded its time limit')
        available = np.flatnonzero(a[column:, column])
        if not len(available):
            return 0
        pivot = column + int(available[0])
        if pivot != column:
            a[[column, pivot]] = a[[pivot, column]]
            answer = -answer % prime
        value = int(a[column, column])
        answer = answer * value % prime
        a[column, column:] = a[column, column:] * pow(value, -1, prime) % prime
        if column + 1 < len(a):
            a[column+1:, column:] = (
                a[column+1:, column:]
                - a[column+1:, column:column+1] * a[column:column+1, column:]
            ) % prime
    return int(answer)


def column_rank(matrix, prime, limit):
    start = time.monotonic()
    a = np.array(matrix, dtype=np.int64, copy=True) % prime
    rank = 0
    for column in range(a.shape[1]):
        if time.monotonic() - start > limit:
            raise TimeoutError('Modular rank phase exceeded its time limit')
        available = np.flatnonzero(a[rank:, column])
        if not len(available):
            continue
        pivot = rank + int(available[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank, column:] = a[rank, column:] * pow(int(a[rank, column]), -1, prime) % prime
        if rank + 1 < len(a):
            a[rank+1:, column:] = (
                a[rank+1:, column:]
                - a[rank+1:, column:column+1] * a[rank:rank+1, column:]
            ) % prime
        rank += 1
        if rank == min(a.shape):
            break
    return rank


def closed_form(point, raw_input, selections, prime):
    """Independent manual derivative, separate from reverse-mode AD."""
    P, M, A, A3, head = point
    x = P @ raw_input % prime
    U, V = selections[1], selections[2]
    w = head[0]
    B = w[:, None] * A3 % prime
    Mx = M @ x % prime
    H1 = x * Mx % prime
    AH1 = A @ H1 % prime
    ell = B.T @ x[V] % prime
    L2 = x[U] * ell % prime
    suffix = A.T @ L2 % prime
    K = x * suffix % prime
    H2 = x[U] * AH1 % prime
    gradient_x = (Mx * suffix + M.T @ K) % prime
    np.add.at(gradient_x, U, AH1 * ell % prime)
    np.add.at(gradient_x, V, B @ H2 % prime)
    gradient_x %= prime
    matrices = [np.outer(gradient_x, raw_input), np.outer(K, x),
                np.outer(L2, H1), np.outer(w*x[V] % prime, H2),
                (x[V] * (A3 @ H2 % prime) % prime)[None, :]]
    derivative = np.concatenate([block.ravel() for block in matrices]) % prime
    value = int(L2 @ AH1 % prime)
    quadrics = np.concatenate([H1, L2]) % prime
    cubics = np.concatenate([K, gradient_x, H2]) % prime
    return value, derivative, quadrics, cubics


def check_minor(matrix, witness, prime, limit):
    rows, columns = witness['rows_zero_based'], witness['columns_zero_based']
    assert len(rows) == len(set(rows)) == witness['size']
    assert len(columns) == len(set(columns)) == witness['size']
    assert min(rows) >= 0 and max(rows) < matrix.shape[0]
    assert min(columns) >= 0 and max(columns) < matrix.shape[1]
    actual = determinant(matrix[np.ix_(rows, columns)], prime, limit)
    assert actual == witness['determinant_mod_prime'] != 0
    return actual


def verify_case(path, limit):
    started = time.monotonic()
    c = json.loads(path.read_text())
    a, p, W = c['block_size'], c['prime'], c['input_dimension']
    assert a in [3, 4, 5] and W == 4*a
    assert p == 1000003 and all(p % j for j in range(2, int(p**0.5)+1))
    widths = c['crossing_widths']
    assert widths == [4*a, 2*a, a]
    integer_point = [np.asarray(block, dtype=np.int64) for block in c['integer_parameter_point']]
    assert [list(block.shape) for block in integer_point] == [[W,W], [W,W], [2*a,W], [a,2*a], [1,a]]
    assert np.array_equal(integer_point[0], np.eye(W, dtype=np.int64))
    assert np.all(integer_point[4] == 1)
    assert array_hash(np.concatenate([block.ravel() for block in integer_point])) == c['integer_parameter_point_int64_little_endian_sha256']
    point = [block % p for block in integer_point]
    selections = network.balanced(W, widths)
    assert [s.tolist() for s in selections] == c['selections_zero_based']
    X = np.asarray(c['input_points'], dtype=np.int64)
    assert X.ndim == 2 and X.shape[1] == W and np.all((X >= 0) & (X < p))
    assert array_hash(X) == c['expected_int64_little_endian_sha256']['input_points']
    parameter_count = sum(block.size for block in point)
    # Bounds every unreduced matrix product/sum used by this verifier.
    assert 2*parameter_count*p*p < np.iinfo(np.int64).max
    flat = np.concatenate([block.ravel() for block in point])
    offsets = np.cumsum([0] + [block.size for block in point])
    values, gradients, quadrics, cubics = [], [], [], []
    Euler_checks = 0
    for raw_input in X:
        value, derivative, input_gradient = network.direct_value_and_gradient(point, raw_input, selections, p)
        other_value, other_derivative, q, cubic = closed_form(point, raw_input, selections, p)
        assert value == other_value
        assert np.array_equal(derivative, other_derivative)
        values.append(value)
        gradients.append(derivative)
        if len(quadrics) < 10*a+12:
            quadrics.append(q)
            cubics.append(cubic)
        for block, degree in enumerate([4,1,1,1,1]):
            lo, hi = offsets[block:block+2]
            assert int(derivative[lo:hi] @ flat[lo:hi] % p) == degree*value % p
            Euler_checks += 1
        assert int(input_gradient @ raw_input % p) == 4*value % p
        Euler_checks += 1
    J, Q, C = np.asarray(gradients), np.asarray(quadrics), np.asarray(cubics)
    G = network.gauge_tangents(point, selections, p)
    for name, matrix in [('J',J), ('Q',Q), ('C',C), ('gauge',G)]:
        assert list(matrix.shape) == c['expected_shapes'][name]
        assert array_hash(matrix) == c['expected_int64_little_endian_sha256'][name]
    assert np.all(J @ G % p == 0)
    assert column_rank(G, p, limit) == c['gauge_rank'] == 11*a
    assert c['minors']['jointP']['size'] == 42*a*a-10*a
    assert c['minors']['quadratic_bank']['size'] == 6*a
    assert c['minors']['cubic_bank']['size'] == 10*a
    assert max(c['minors']['jointP']['columns_zero_based']) < c['parameter_count_without_head'] == 42*a*a
    determinants = {}
    for name, matrix in [('jointP',J), ('quadratic_bank',Q), ('cubic_bank',C)]:
        determinants[name] = check_minor(matrix, c['minors'][name], p, limit)
    finite_checks = 0
    for row in range(8):
        for block in range(1,5):
            coordinate = (row*17 + block*11) % point[block].size
            changed = [b.copy() for b in point]
            changed[block].ravel()[coordinate] += 1
            changed[block] %= p
            value, _, _ = network.direct_value_and_gradient(changed, X[row], selections, p)
            assert (value-values[row]) % p == int(J[row, offsets[block]+coordinate])
            finite_checks += 1
        coordinate = (row*17+11) % point[0].size
        interpolated = 0
        for step, weight in enumerate([-25,48,-36,16,-3]):
            changed = [b.copy() for b in point]
            changed[0].ravel()[coordinate] += step
            changed[0] %= p
            value, _, _ = network.direct_value_and_gradient(changed, X[row], selections, p)
            interpolated += weight*value
        assert interpolated*pow(12,-1,p) % p == int(J[row,coordinate])
        finite_checks += 1
    return {'certificate':path.name, 'block_size':a,
            'jointP_rank':42*a*a-10*a, 'quadratic_bank_rank':6*a, 'cubic_bank_rank':10*a,
            'nonzero_minor_determinants_mod_prime':determinants,
            'gauge_rank':11*a, 'independent_derivative_rows_compared':len(X),
            'Euler_identities_verified':Euler_checks,
            'scalar_finite_difference_or_interpolation_checks':finite_checks,
            'seconds':round(time.monotonic()-started,3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-phase-seconds', type=float, default=30)
    args = parser.parse_args()
    assert args.max_phase_seconds > 0
    files = check_distribution_hashes()
    cases = [verify_case(ROOT/'certificates'/f'a{a}.json', args.max_phase_seconds) for a in [3,4,5]]
    print(json.dumps({'status':'PASS','distribution_file_hashes_checked':files,
                      'exact_nonzero_minors_verified':9,'cases':cases},indent=2))


if __name__ == '__main__':
    main()

"""Exact finite-field network evaluation and reverse-mode differentiation.

This standalone arithmetic module has no filesystem or network dependencies.
"""

import numpy as np


def balanced(width, dimensions):
    uses = [0] * width
    result = []
    for dimension in dimensions:
        chosen = []
        while len(chosen) < dimension:
            minimum = min(uses)
            candidates = [j for j, count in enumerate(uses) if count == minimum]
            remaining = dimension - len(chosen)
            if remaining >= len(candidates):
                take = candidates
            else:
                take = [candidates[(j * len(candidates)) // remaining]
                        for j in range(remaining)]
            for j in take:
                uses[j] += 1
            chosen.extend(take)
        result.append(np.array(sorted(chosen), dtype=np.int64))
    return result


def direct_value_and_gradient(point, vector, selections, prime):
    """Scalar network forward evaluation and reverse-mode parameter gradient."""
    P, A1, A2, A3, head = point
    matrices = [A1, A2, A3]
    x = P @ vector % prime
    states = [x]
    gates, transformed = [], []
    for A, indices in zip(matrices, selections):
        gate = x[indices]
        linear = A @ states[-1] % prime
        gates.append(gate)
        transformed.append(linear)
        states.append(gate * linear % prime)
    w = head[0]
    value = int(w @ states[-1] % prime)
    grad_matrices = [None] * 3
    adjoint_state = w.copy()
    adjoint_input_gates = np.zeros_like(x)
    for layer in reversed(range(3)):
        adjoint_linear = adjoint_state * gates[layer] % prime
        grad_matrices[layer] = np.outer(adjoint_linear, states[layer]) % prime
        adjoint_gate = adjoint_state * transformed[layer] % prime
        # np.add.at also handles potential repeated selector entries correctly.
        np.add.at(adjoint_input_gates, selections[layer], adjoint_gate)
        adjoint_input_gates %= prime
        adjoint_state = matrices[layer].T @ adjoint_linear % prime
    adjoint_x = (adjoint_state + adjoint_input_gates) % prime
    gradients = [np.outer(adjoint_x, vector) % prime,
                 *grad_matrices, states[-1][None, :]]
    gradient = np.concatenate([block.ravel() for block in gradients]) % prime
    input_gradient = P.T @ adjoint_x % prime
    return value, gradient, input_gradient


def flatten(blocks, prime):
    return np.concatenate([block.ravel() for block in blocks]) % prime


def gauge_tangents(point, selections, prime):
    P, A1, A2, A3, head = point
    matrices = [A1, A2, A3]
    tangents = []
    # Hidden-state diagonal scalings.
    for layer, A in enumerate(matrices):
        for channel in range(A.shape[0]):
            blocks = [np.zeros_like(block) for block in point]
            blocks[layer + 1][channel, :] += A[channel, :]
            if layer < 2:
                blocks[layer + 2][:, channel] -= matrices[layer + 1][:, channel]
            else:
                blocks[4][0, channel] -= head[0, channel]
            tangents.append(flatten(blocks, prime))
    # Projected-input row scaling; compensate both input to A1 and all gates.
    for coordinate in range(P.shape[0]):
        blocks = [np.zeros_like(block) for block in point]
        blocks[0][coordinate, :] = P[coordinate, :]
        blocks[1][:, coordinate] -= A1[:, coordinate]
        for layer, A in enumerate(matrices):
            for row, selected in enumerate(selections[layer]):
                if selected == coordinate:
                    blocks[layer + 1][row, :] -= A[row, :]
        tangents.append(flatten(blocks, prime))
    return np.array(tangents, dtype=np.int64).T

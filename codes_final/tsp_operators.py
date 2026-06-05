
from __future__ import annotations

import numpy as np
import random
from collections.abc import Callable


def two_opt_best(tour: list[int], dist_matrix: np.ndarray) -> list[int]:
    n = len(tour)
    if n < 2:                          # O3: guard for degenerate tour
        return tour[:]

    num_tries = min(n * 2, 100)
    best_delta = 0.0   # O1: only strictly negative (improving) deltas win
    best_i = -1
    best_j = -1

    for _ in range(num_tries):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)

        # O5: Delta cost -- only 4 edges are affected by a 2-opt swap
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (dist_matrix[a, c] + dist_matrix[b, d]
                 - dist_matrix[a, b] - dist_matrix[c, d])

        if delta < best_delta:         # strictly better than the current best
            best_delta = delta
            best_i, best_j = i, j

    if best_i >= 0:                    # at least one improving move was found
        new_tour = tour[:]
        new_tour[best_i + 1:best_j + 1] = new_tour[best_i + 1:best_j + 1][::-1]  # O2 & Fix: correct slice match for delta
        return new_tour
    return tour[:]                     # no improvement found


def swap_operator(tour: list[int], dist_matrix: np.ndarray) -> list[int]:
    n = len(tour)
    if n < 2:                          # O3: guard
        return tour[:]
    new_tour = tour[:]
    i, j = random.sample(range(n), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def relocate_operator(tour: list[int], dist_matrix: np.ndarray) -> list[int]:
    n = len(tour)
    if n < 3:
        return tour[:]
    new_tour = tour[:]
    i = random.randint(0, n - 1)
    city = new_tour.pop(i)
    # Pick from n-1 positions that are NOT i
    j = random.randint(0, n - 2)
    if j >= i:
        j += 1           # shift up to skip original index i
    new_tour.insert(j, city)
    return new_tour


def restart_operator(tour: list[int], _dist_matrix: np.ndarray) -> list[int]:
    n = len(tour)
    new_tour = list(range(n))
    random.shuffle(new_tour)
    return new_tour


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------
OPERATORS: dict[int, tuple[str, Callable]] = {
    0: ('2-opt',    two_opt_best),
    1: ('swap',     swap_operator),
    2: ('relocate', relocate_operator),
    3: ('restart',  restart_operator),
}

OPERATOR_NAMES: list[str] = [name for _, (name, _) in sorted(OPERATORS.items())]
NUM_OPERATORS: int = len(OPERATORS)

"""
TSP Local Search Operators Module
Implements: 2-opt, swap, relocate, and restart operators.
"""

import numpy as np
import random
from tsp_data import tour_length


def two_opt(tour, dist_matrix):
    """Apply a single random 2-opt move (reverse a sub-segment)."""
    n = len(tour)
    new_tour = tour[:]
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour


def two_opt_best(tour, dist_matrix):
    """Apply best-improvement 2-opt: try many random moves, keep the best."""
    best_tour = tour[:]
    best_len = tour_length(tour, dist_matrix)
    n = len(tour)
    # Try a limited number of random 2-opt moves for efficiency
    num_tries = min(n * 2, 100)
    for _ in range(num_tries):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        candidate = tour[:]
        candidate[i:j+1] = reversed(candidate[i:j+1])
        cand_len = tour_length(candidate, dist_matrix)
        if cand_len < best_len:
            best_tour = candidate
            best_len = cand_len
    return best_tour


def swap_operator(tour, dist_matrix):
    """Swap two randomly selected cities in the tour."""
    n = len(tour)
    new_tour = tour[:]
    i, j = random.sample(range(n), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def relocate_operator(tour, dist_matrix):
    """Remove a city and re-insert it at a different random position.

    Fix (M3): After pop(i), valid positions are 0..n-1. We pick
    j in [0, n-2] and shift by 1 if j >= i so j never equals i,
    guaranteeing the city moves to a genuinely different location.
    """
    n = len(tour)
    if n < 3:
        return tour[:]
    new_tour = tour[:]
    i = random.randint(0, n - 1)
    city = new_tour.pop(i)
    # Pick from n-1 positions that are NOT i
    j = random.randint(0, n - 2)
    if j >= i:
        j += 1           # shift up to skip index i
    new_tour.insert(j, city)
    return new_tour


def restart_operator(tour, dist_matrix):
    """Generate a random permutation (restart from scratch)."""
    n = len(tour)
    new_tour = list(range(n))
    random.shuffle(new_tour)
    return new_tour


# Operator registry
OPERATORS = {
    0: ('2-opt', two_opt_best),
    1: ('swap', swap_operator),
    2: ('relocate', relocate_operator),
    3: ('restart', restart_operator),
}

OPERATOR_NAMES = ['2-opt', 'swap', 'relocate', 'restart']
NUM_OPERATORS = len(OPERATORS)

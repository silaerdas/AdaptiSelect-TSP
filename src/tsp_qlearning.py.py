"""
Q-Learning Agent for Adaptive Operator Selection in TSP.
Models the search process as an MDP with tabular Q-learning.
"""

import numpy as np
import random
from tsp_data import tour_length, nearest_neighbor_tour
from tsp_operators import OPERATORS, NUM_OPERATORS, OPERATOR_NAMES


# --------------- State Discretization ---------------

def get_quality_band(current_length, optimal, num_bands=5):
    """Discretize solution quality into bands based on gap to optimal."""
    gap = (current_length - optimal) / optimal  # relative gap
    if gap < 0.05:
        return 0  # excellent (<5%)
    elif gap < 0.15:
        return 1  # good (5-15%)
    elif gap < 0.30:
        return 2  # fair (15-30%)
    elif gap < 0.50:
        return 3  # poor (30-50%)
    else:
        return 4  # very poor (>50%)


def get_iteration_stage(iteration, max_iterations, num_stages=3):
    """Discretize iteration into early/mid/late stages."""
    ratio = iteration / max_iterations
    if ratio < 0.33:
        return 0  # early
    elif ratio < 0.66:
        return 1  # mid
    else:
        return 2  # late


def encode_state(quality_band, improved, iteration_stage, last_operator):
    """Encode state tuple into a single integer index."""
    # quality_band: 0-4, improved: 0-1, iteration_stage: 0-2, last_operator: 0-3
    return (quality_band * 2 * 3 * 4 +
            improved * 3 * 4 +
            iteration_stage * 4 +
            last_operator)


NUM_QUALITY_BANDS = 5
NUM_STAGES = 3
NUM_STATES = NUM_QUALITY_BANDS * 2 * NUM_STAGES * NUM_OPERATORS  # 120


# --------------- Reward Functions ---------------

def reward_pure_improvement(prev_length, new_length, global_best, **kwargs):
    """R1: Pure improvement reward."""
    return prev_length - new_length


def reward_global_bonus(prev_length, new_length, global_best, bonus=100.0, **kwargs):
    """R2: Improvement + bonus when new global best is found."""
    r = prev_length - new_length
    if new_length < global_best:
        r += bonus
    return r


# --------------- Epsilon Strategies ---------------

class ConstantEpsilon:
    """Fixed exploration rate."""
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.name = f"Constant (ε={epsilon})"

    def get_epsilon(self):
        return self.epsilon

    def update(self, improved):
        pass

    def get_history(self):
        return []


class ExponentialDecayEpsilon:
    """Exponential decay of exploration rate."""
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.name = "Exponential Decay"
        self.history = []

    def get_epsilon(self):
        return self.epsilon

    def update(self, improved):
        self.history.append(self.epsilon)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def get_history(self):
        return self.history


class AdaptiveEpsilon:
    """Adaptive epsilon with stagnation detection."""
    def __init__(self, epsilon_init=0.1, epsilon_min=0.01, epsilon_max=0.8,
                 stagnation_threshold=50, increase_rate=0.05, decrease_rate=0.99):
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.stagnation_threshold = stagnation_threshold
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.no_improve_count = 0
        self.name = "Adaptive (Stagnation)"
        self.history = []

    def get_epsilon(self):
        return self.epsilon

    def update(self, improved):
        self.history.append(self.epsilon)
        if improved:
            self.no_improve_count = 0
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decrease_rate)
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.stagnation_threshold:
                self.epsilon = min(self.epsilon_max,
                                   self.epsilon + self.increase_rate)

    def get_history(self):
        return self.history


# --------------- Q-Learning Agent ---------------

class QLearningAgent:
    """Tabular Q-Learning agent for operator selection."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon_strategy=None,
                 reward_fn=None, reward_name="R1"):
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((NUM_STATES, NUM_OPERATORS))
        self.epsilon_strategy = epsilon_strategy or ConstantEpsilon(0.1)
        self.reward_fn = reward_fn or reward_pure_improvement
        self.reward_name = reward_name

    def select_action(self):
        """Epsilon-greedy action selection."""
        eps = self.epsilon_strategy.get_epsilon()
        if random.random() < eps:
            return random.randint(0, NUM_OPERATORS - 1)
        else:
            return int(np.argmax(self.q_table[self.state]))

    def update(self, state, action, reward, next_state):
        """Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def run_episode(self, dist_matrix, optimal, max_iterations=2000, seed=None):
        """Run one episode of Q-learning guided local search."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = dist_matrix.shape[0]

        # Initial solution via nearest neighbor
        tour = nearest_neighbor_tour(dist_matrix, start=random.randint(0, n-1))
        current_length = tour_length(tour, dist_matrix)
        global_best_length = current_length
        best_tour = tour[:]

        # Initial state
        last_operator = 0
        improved = 0
        quality_band = get_quality_band(current_length, optimal)
        iteration_stage = get_iteration_stage(0, max_iterations)
        self.state = encode_state(quality_band, improved, iteration_stage, last_operator)

        # Tracking
        length_history = [current_length]
        operator_history = []
        transition_counts = np.zeros((NUM_OPERATORS, NUM_OPERATORS))
        restart_count = 0
        last_successful_op = None

        for iteration in range(max_iterations):
            # Select action
            action = self.select_action()
            operator_history.append(action)

            # Apply operator
            _, op_func = OPERATORS[action]
            new_tour = op_func(tour, dist_matrix)
            new_length = tour_length(new_tour, dist_matrix)

            # Compute reward
            reward = self.reward_fn(current_length, new_length, global_best_length)

            # Accept move (always accept — let RL learn)
            improved_flag = 1 if new_length < current_length else 0

            # Track transitions after successful moves
            if improved_flag and last_successful_op is not None:
                transition_counts[last_successful_op, action] += 1
            if improved_flag:
                last_successful_op = action

            if action == 3:
                restart_count += 1

            # Update best
            if new_length < global_best_length:
                global_best_length = new_length
                best_tour = new_tour[:]

            # Accept improving moves; for worsening, accept with small probability
            if new_length < current_length:
                tour = new_tour
                current_length = new_length
            elif random.random() < 0.01:  # small probability to accept worse
                tour = new_tour
                current_length = new_length

            # Next state
            quality_band = get_quality_band(current_length, optimal)
            iteration_stage = get_iteration_stage(iteration + 1, max_iterations)
            next_state = encode_state(quality_band, improved_flag, iteration_stage, action)

            # Q-learning update
            self.update(self.state, action, reward, next_state)
            self.state = next_state

            # Update epsilon
            self.epsilon_strategy.update(improved_flag == 1)

            length_history.append(global_best_length)

        results = {
            'best_length': global_best_length,
            'best_tour': best_tour,
            'length_history': length_history,
            'operator_history': operator_history,
            'transition_counts': transition_counts,
            'restart_count': restart_count,
            'gap': (global_best_length - optimal) / optimal * 100,
        }
        return results


# --------------- Baseline Strategies ---------------

def run_fixed_operator(op_id, dist_matrix, optimal, max_iterations=2000, seed=None):
    """Run a fixed single operator for all iterations."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = dist_matrix.shape[0]
    tour = nearest_neighbor_tour(dist_matrix, start=random.randint(0, n-1))
    current_length = tour_length(tour, dist_matrix)
    global_best = current_length
    best_tour = tour[:]
    length_history = [current_length]

    _, op_func = OPERATORS[op_id]

    for _ in range(max_iterations):
        new_tour = op_func(tour, dist_matrix)
        new_length = tour_length(new_tour, dist_matrix)

        if new_length < current_length:
            tour = new_tour
            current_length = new_length
        elif random.random() < 0.01:
            tour = new_tour
            current_length = new_length

        if current_length < global_best:
            global_best = current_length
            best_tour = tour[:]

        length_history.append(global_best)

    return {
        'best_length': global_best,
        'best_tour': best_tour,
        'length_history': length_history,
        'gap': (global_best - optimal) / optimal * 100,
    }


def run_random_operator(dist_matrix, optimal, max_iterations=2000, seed=None):
    """Run random operator selection for all iterations."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = dist_matrix.shape[0]
    tour = nearest_neighbor_tour(dist_matrix, start=random.randint(0, n-1))
    current_length = tour_length(tour, dist_matrix)
    global_best = current_length
    best_tour = tour[:]
    length_history = [current_length]

    for _ in range(max_iterations):
        action = random.randint(0, NUM_OPERATORS - 1)
        _, op_func = OPERATORS[action]
        new_tour = op_func(tour, dist_matrix)
        new_length = tour_length(new_tour, dist_matrix)

        if new_length < current_length:
            tour = new_tour
            current_length = new_length
        elif random.random() < 0.01:
            tour = new_tour
            current_length = new_length

        if current_length < global_best:
            global_best = current_length
            best_tour = tour[:]

        length_history.append(global_best)

    return {
        'best_length': global_best,
        'best_tour': best_tour,
        'length_history': length_history,
        'gap': (global_best - optimal) / optimal * 100,
    }

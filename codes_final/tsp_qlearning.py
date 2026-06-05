
from __future__ import annotations

import numpy as np
import random
from tsp_data import tour_length, nearest_neighbor_tour
from tsp_operators import OPERATORS, NUM_OPERATORS


# --------------- State Discretization ---------------

def get_quality_band(current_length: float, optimal: float,
                     num_bands: int = 5) -> int:
  
    if optimal <= 0:
        raise ValueError(f"optimal must be positive, got {optimal}")
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


def get_iteration_stage(iteration: int, max_iterations: int,
                        num_stages: int = 3) -> int:
    if max_iterations <= 0:
        raise ValueError(
            f"max_iterations must be positive, got {max_iterations}"
        )
    ratio = iteration / max_iterations
    if ratio < 0.33:
        return 0  # early
    elif ratio < 0.66:
        return 1  # mid
    else:
        return 2  # late


def encode_state(quality_band: int, improved: int,
  
    return (quality_band * 2 * 3 * 4 +
            improved * 3 * 4 +
            iteration_stage * 4 +
            last_operator)


NUM_QUALITY_BANDS = 5
NUM_STAGES = 3
NUM_STATES = NUM_QUALITY_BANDS * 2 * NUM_STAGES * NUM_OPERATORS  # 120


# --------------- Reward Functions ---------------

def reward_pure_improvement(prev_length: float, new_length: float,
                             global_best: float, **kwargs) -> float:

    return prev_length - new_length


def reward_global_bonus(prev_length: float, new_length: float,
                        global_best: float, bonus: float = 100.0,
                        **kwargs) -> float:
    r = prev_length - new_length
    if new_length < global_best:
        r += bonus
    return r


# --------------- Epsilon Strategies ---------------

class ConstantEpsilon:
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.name = f"Constant (ε={epsilon})"

    def get_epsilon(self) -> float:
        return self.epsilon

    def update(self, improved: bool) -> None:
        pass

    def get_history(self) -> list[float]:
        return []


class ExponentialDecayEpsilon:
    def __init__(self, epsilon_start: float = 1.0,
                 epsilon_min: float = 0.01,
                 decay: float = 0.995) -> None:
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.name = "Exponential Decay"
        self.history: list[float] = []

    def get_epsilon(self) -> float:
        return self.epsilon

    def update(self, improved: bool) -> None:
        self.history.append(self.epsilon)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def get_history(self) -> list[float]:
        return self.history


class AdaptiveEpsilon:
  
    def __init__(self, epsilon_init: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_max: float = 0.8,
                 stagnation_threshold: int = 50,
                 increase_rate: float = 0.05,
                 decrease_rate: float = 0.99) -> None:
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.stagnation_threshold = stagnation_threshold
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.no_improve_count = 0
        self.name = "Adaptive (Stagnation)"
        self.history: list[float] = []

    def get_epsilon(self) -> float:
        return self.epsilon

    def update(self, improved: bool) -> None:
        self.history.append(self.epsilon)
        if improved:
            self.no_improve_count = 0
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.decrease_rate)
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.stagnation_threshold:
                self.epsilon = min(self.epsilon_max,
                                   self.epsilon + self.increase_rate)
                self.no_improve_count = 0   # Q5: reset counter after boost

    def get_history(self) -> list[float]:
        return self.history


# --------------- Q-Learning Agent ---------------

class QLearningAgent:
    """tabular Q-Learning agent for operator selection.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon_strategy=None,
                 reward_fn=None, reward_name: str = "R1") -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((NUM_STATES, NUM_OPERATORS))
        self.epsilon_strategy = epsilon_strategy or ConstantEpsilon(0.1)
        self.reward_fn = reward_fn or reward_pure_improvement
        self.reward_name = reward_name
        self.state: int = 0  # safe default; overwritten by run_episode()

    def select_action(self, state: int | None = None) -> int:
        """Epsilon-greedy action selection.
        """
        s = state if state is not None else self.state
        eps = self.epsilon_strategy.get_epsilon()
        if random.random() < eps:
            return random.randint(0, NUM_OPERATORS - 1)
        else:
            return int(np.argmax(self.q_table[s]))

    def update_q_value(self, state: int, action: int, reward: float,
                       next_state: int) -> None:
        """Q-learning update rule (Bellman equation)."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    # Keep the old name as an alias for backward compatibility
    update = update_q_value

    def run_episode(self, dist_matrix: np.ndarray, optimal: float,
                    max_iterations: int = 2000,
                    seed: int | None = None) -> dict:
  
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = dist_matrix.shape[0]

        # Initial solution via nearest neighbour
        tour = nearest_neighbor_tour(dist_matrix,
                                     start=random.randint(0, n - 1))
        current_length = tour_length(tour, dist_matrix)
        global_best_length = current_length
        best_tour = tour[:]

        # Tracking arrays
        length_history = [current_length]
        operator_counts = np.zeros(NUM_OPERATORS, dtype=int)
        operator_success = np.zeros(NUM_OPERATORS, dtype=int)
        transition_counts = np.zeros((NUM_OPERATORS, NUM_OPERATORS), dtype=int)
        improve_counts = np.zeros(NUM_OPERATORS, dtype=int)
        restart_count = 0

        last_operator = 0
       
        last_improving_op: int | None = None
        quality_band = get_quality_band(current_length, optimal)
        improved_flag = 0
        iteration_stage = get_iteration_stage(0, max_iterations)
        state = encode_state(quality_band, improved_flag,
                             iteration_stage, last_operator)

        for iteration in range(max_iterations):
            action = self.select_action(state)
            operator_counts[action] += 1
           
            _, op_func = OPERATORS[action]
            new_tour = op_func(tour, dist_matrix)
            new_length = tour_length(new_tour, dist_matrix)

            # ---------------------------------------------------------
            # Consistent baseline: capture pre-update local length
            # ---------------------------------------------------------
            prev_local_length = current_length
            is_locally_better = (new_length < prev_local_length)

            # 1. Reward computation (always uses the pre-update baseline).
          
            reward = self.reward_fn(prev_local_length, new_length,
                                    global_best_length)

            if action == 3:
                restart_count += 1

            # 2. Acceptance logic
            accepted = False
            if action == 3:
                tour = new_tour
                current_length = new_length
                accepted = True
            elif is_locally_better:
                tour = new_tour
                current_length = new_length
                accepted = True
                operator_success[action] += 1
                improve_counts[action] += 1
                # Record the improving-move -> improving-move transition
                # (restart is handled above and is therefore excluded here).
                if last_improving_op is not None:
                    transition_counts[last_improving_op, action] += 1
                last_improving_op = action
            elif random.random() < 0.01:  # SA-like acceptance of worse move
                tour = new_tour
                current_length = new_length
                accepted = True

            # 3. Global best tracking
            global_improved = False
            if current_length < global_best_length:
                global_best_length = current_length
                best_tour = tour[:]
                global_improved = True

            self.epsilon_strategy.update(bool(is_locally_better))

            # 4. Next-state transition flag
            # Encodes whether the operator physically improved the tour
            next_improved_flag = 1 if (accepted and is_locally_better) else 0

            next_quality_band = get_quality_band(current_length, optimal)
            next_iteration_stage = get_iteration_stage(iteration + 1,
                                                       max_iterations)
            next_state = encode_state(next_quality_band, next_improved_flag,
                                      next_iteration_stage, action)

            # 5. Q-learning (Bellman) update
            self.update_q_value(state, action, reward, next_state)

            state = next_state
            last_operator = action
            length_history.append(global_best_length)  # best-so-far for plots

        gap = ((global_best_length - optimal) / optimal * 100
               if optimal > 0 else float('inf'))
        return {
            'best_length': global_best_length,
            'best_tour': best_tour,
            'length_history': length_history,
            'operator_counts': operator_counts,
            'operator_success': operator_success,
            'transition_counts': transition_counts,
            'improve_counts': improve_counts,
            'restart_count': restart_count,
            'gap': gap,
        }


# --------------- Baseline Strategies ---------------

def run_fixed_operator(op_id: int, dist_matrix: np.ndarray,
                       optimal: float, max_iterations: int = 2000,
                       seed: int | None = None) -> dict:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = dist_matrix.shape[0]
    tour = nearest_neighbor_tour(dist_matrix,
                                 start=random.randint(0, n - 1))
    current_length = tour_length(tour, dist_matrix)
    global_best = current_length
    best_tour = tour[:]
    length_history = [current_length]

    _, op_func = OPERATORS[op_id]

    for _ in range(max_iterations):
        new_tour = op_func(tour, dist_matrix)
        new_length = tour_length(new_tour, dist_matrix)

        # Q4: First decide acceptance, then update global best
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


def run_random_operator(dist_matrix: np.ndarray, optimal: float,
                        max_iterations: int = 2000,
                        seed: int | None = None) -> dict:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = dist_matrix.shape[0]
    tour = nearest_neighbor_tour(dist_matrix,
                                 start=random.randint(0, n - 1))
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

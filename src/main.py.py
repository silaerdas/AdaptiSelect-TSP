"""
Main Runner: Adaptive Local Search Operator Selection for TSP via Q-Learning
Runs all experiments, generates comparison tables and visualizations.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys
import time

from tsp_data import get_instance, tour_length, OPTIMAL
from tsp_operators import OPERATOR_NAMES, NUM_OPERATORS
from tsp_qlearning import (
    QLearningAgent, ConstantEpsilon, ExponentialDecayEpsilon, AdaptiveEpsilon,
    reward_pure_improvement, reward_global_bonus,
    run_fixed_operator, run_random_operator
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_ITERATIONS = 3000
NUM_RUNS = 5  # average over multiple runs
SEED_BASE = 42

TRAIN_INSTANCES = ['eil51', 'berlin52']
TEST_INSTANCES = ['st70', 'kroA100']
ALL_INSTANCES = TRAIN_INSTANCES + TEST_INSTANCES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Experiment 1: Main comparison (Q-Learning vs Baselines)
# ============================================================

def run_main_comparison():
    print("=" * 70)
    print("EXPERIMENT 1: Q-Learning (Adaptive eps, R1) vs Baselines")
    print("=" * 70)

    all_results = {}

    for inst_name in ALL_INSTANCES:
        print(f"\n--- Instance: {inst_name} ---")
        coords, dist_matrix, optimal = get_instance(inst_name)

        strategies = {}

        # Q-Learning with Adaptive Epsilon + R1
        best_gaps = []
        all_histories = []
        best_transitions = None
        best_gap = float('inf')
        for run in range(NUM_RUNS):
            agent = QLearningAgent(
                alpha=0.1, gamma=0.9,
                epsilon_strategy=AdaptiveEpsilon(),
                reward_fn=reward_pure_improvement,
                reward_name="R1"
            )
            result = agent.run_episode(dist_matrix, optimal,
                                        max_iterations=MAX_ITERATIONS,
                                        seed=SEED_BASE + run)
            best_gaps.append(result['gap'])
            all_histories.append(result['length_history'])
            if result['gap'] < best_gap:
                best_gap = result['gap']
                best_transitions = result['transition_counts']
                best_tour_result = result
        strategies['Q-Learning (Adaptive)'] = {
            'avg_gap': np.mean(best_gaps),
            'std_gap': np.std(best_gaps),
            'best_gap': np.min(best_gaps),
            'avg_history': np.mean(all_histories, axis=0),
            'transitions': best_transitions,
            'best_result': best_tour_result,
        }

        # Fixed operators
        for op_id, op_name in enumerate(OPERATOR_NAMES):
            gaps = []
            histories = []
            for run in range(NUM_RUNS):
                result = run_fixed_operator(op_id, dist_matrix, optimal,
                                             max_iterations=MAX_ITERATIONS,
                                             seed=SEED_BASE + run)
                gaps.append(result['gap'])
                histories.append(result['length_history'])
            strategies[f'Fixed {op_name}'] = {
                'avg_gap': np.mean(gaps),
                'std_gap': np.std(gaps),
                'best_gap': np.min(gaps),
                'avg_history': np.mean(histories, axis=0),
            }

        # Random selection
        gaps = []
        histories = []
        for run in range(NUM_RUNS):
            result = run_random_operator(dist_matrix, optimal,
                                          max_iterations=MAX_ITERATIONS,
                                          seed=SEED_BASE + run)
            gaps.append(result['gap'])
            histories.append(result['length_history'])
        strategies['Random'] = {
            'avg_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'best_gap': np.min(gaps),
            'avg_history': np.mean(histories, axis=0),
        }

        all_results[inst_name] = strategies

        # Print results for this instance
        print(f"  {'Strategy':<25} {'Avg Gap%':>10} {'Std':>8} {'Best Gap%':>10}")
        print(f"  {'-'*55}")
        for name, data in strategies.items():
            print(f"  {name:<25} {data['avg_gap']:>10.2f} {data['std_gap']:>8.2f} {data['best_gap']:>10.2f}")

    return all_results


# ============================================================
# Experiment 2: Reward Shaping Comparison (R1 vs R2)
# ============================================================

def run_reward_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Reward Shaping Comparison (R1 vs R2)")
    print("=" * 70)

    reward_results = {}

    for inst_name in TRAIN_INSTANCES:
        print(f"\n--- Instance: {inst_name} ---")
        coords, dist_matrix, optimal = get_instance(inst_name)

        for rname, rfn in [("R1 (Pure)", reward_pure_improvement),
                            ("R2 (Global Bonus)", reward_global_bonus)]:
            gaps = []
            histories = []
            restart_counts = []
            for run in range(NUM_RUNS):
                agent = QLearningAgent(
                    alpha=0.1, gamma=0.9,
                    epsilon_strategy=AdaptiveEpsilon(),
                    reward_fn=rfn, reward_name=rname
                )
                result = agent.run_episode(dist_matrix, optimal,
                                            max_iterations=MAX_ITERATIONS,
                                            seed=SEED_BASE + run)
                gaps.append(result['gap'])
                histories.append(result['length_history'])
                restart_counts.append(result['restart_count'])

            key = f"{inst_name}_{rname}"
            reward_results[key] = {
                'avg_gap': np.mean(gaps),
                'std_gap': np.std(gaps),
                'avg_history': np.mean(histories, axis=0),
                'avg_restarts': np.mean(restart_counts),
                'inst': inst_name,
                'reward': rname,
            }
            print(f"  {rname:<25} Gap: {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}  "
                  f"Restarts: {np.mean(restart_counts):.1f}")

    return reward_results


# ============================================================
# Experiment 3: Epsilon Strategy Comparison
# ============================================================

def run_epsilon_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Epsilon Strategy Comparison")
    print("=" * 70)

    epsilon_results = {}

    for inst_name in TRAIN_INSTANCES:
        print(f"\n--- Instance: {inst_name} ---")
        coords, dist_matrix, optimal = get_instance(inst_name)

        eps_strategies = [
            ("Constant eps=0.1", ConstantEpsilon(0.1)),
            ("Exponential Decay", ExponentialDecayEpsilon(1.0, 0.01, 0.998)),
            ("Adaptive (Stagnation)", AdaptiveEpsilon()),
        ]

        for ename, eps_factory in eps_strategies:
            gaps = []
            histories = []
            restart_counts = []
            eps_histories = []
            for run in range(NUM_RUNS):
                # Create fresh epsilon strategy for each run
                if ename == "Constant eps=0.1":
                    eps = ConstantEpsilon(0.1)
                elif ename == "Exponential Decay":
                    eps = ExponentialDecayEpsilon(1.0, 0.01, 0.998)
                else:
                    eps = AdaptiveEpsilon()

                agent = QLearningAgent(
                    alpha=0.1, gamma=0.9,
                    epsilon_strategy=eps,
                    reward_fn=reward_pure_improvement,
                )
                result = agent.run_episode(dist_matrix, optimal,
                                            max_iterations=MAX_ITERATIONS,
                                            seed=SEED_BASE + run)
                gaps.append(result['gap'])
                histories.append(result['length_history'])
                restart_counts.append(result['restart_count'])
                eps_histories.append(eps.get_history())

            key = f"{inst_name}_{ename}"
            epsilon_results[key] = {
                'avg_gap': np.mean(gaps),
                'std_gap': np.std(gaps),
                'avg_history': np.mean(histories, axis=0),
                'avg_restarts': np.mean(restart_counts),
                'eps_history': eps_histories[0] if eps_histories[0] else [0.1]*MAX_ITERATIONS,
                'inst': inst_name,
                'strategy': ename,
            }
            print(f"  {ename:<25} Gap: {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}  "
                  f"Restarts: {np.mean(restart_counts):.1f}")

    return epsilon_results


# ============================================================
# Visualization Functions
# ============================================================

def plot_convergence(all_results):
    """Plot convergence curves for all strategies on each instance."""
    for inst_name, strategies in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

        for (name, data), color in zip(strategies.items(), colors):
            ax.plot(data['avg_history'], label=name, color=color, linewidth=1.5)

        ax.axhline(y=OPTIMAL[inst_name], color='red', linestyle='--',
                    linewidth=1.5, label=f'Optimal ({OPTIMAL[inst_name]})')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Tour Length', fontsize=12)
        ax.set_title(f'Convergence Comparison — {inst_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'convergence_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: convergence_{inst_name}.png")


def plot_transition_heatmap(all_results):
    """Plot operator transition heatmaps for Q-Learning results."""
    for inst_name, strategies in all_results.items():
        ql_data = strategies.get('Q-Learning (Adaptive)')
        if ql_data is None or ql_data.get('transitions') is None:
            continue

        transitions = ql_data['transitions']
        # Normalize rows to get probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        trans_probs = transitions / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw counts
        sns.heatmap(transitions, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=OPERATOR_NAMES, yticklabels=OPERATOR_NAMES,
                    ax=axes[0])
        axes[0].set_title(f'Transition Counts — {inst_name}', fontweight='bold')
        axes[0].set_xlabel('Next Operator')
        axes[0].set_ylabel('Previous Operator')

        # Probabilities
        sns.heatmap(trans_probs, annot=True, fmt='.2f', cmap='YlGnBu',
                    xticklabels=OPERATOR_NAMES, yticklabels=OPERATOR_NAMES,
                    ax=axes[1])
        axes[1].set_title(f'Transition Probabilities — {inst_name}', fontweight='bold')
        axes[1].set_xlabel('Next Operator')
        axes[1].set_ylabel('Previous Operator')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'transition_heatmap_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: transition_heatmap_{inst_name}.png")


def plot_reward_comparison(reward_results):
    """Plot R1 vs R2 convergence comparison."""
    for inst_name in TRAIN_INSTANCES:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, data in reward_results.items():
            if data['inst'] == inst_name:
                ax.plot(data['avg_history'], label=data['reward'], linewidth=1.5)

        ax.axhline(y=OPTIMAL[inst_name], color='red', linestyle='--',
                    linewidth=1.5, label=f'Optimal ({OPTIMAL[inst_name]})')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Tour Length', fontsize=12)
        ax.set_title(f'Reward Shaping Comparison — {inst_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'reward_comparison_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: reward_comparison_{inst_name}.png")


def plot_epsilon_comparison(epsilon_results):
    """Plot epsilon strategy comparison: convergence + epsilon over time."""
    for inst_name in TRAIN_INSTANCES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Convergence
        for key, data in epsilon_results.items():
            if data['inst'] == inst_name:
                axes[0].plot(data['avg_history'], label=data['strategy'], linewidth=1.5)
        axes[0].axhline(y=OPTIMAL[inst_name], color='red', linestyle='--',
                         linewidth=1.5, label='Optimal')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Tour Length')
        axes[0].set_title(f'Convergence by Epsilon Strategy - {inst_name}', fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Epsilon over time
        for key, data in epsilon_results.items():
            if data['inst'] == inst_name:
                eps_h = data['eps_history']
                if len(eps_h) > 0:
                    axes[1].plot(eps_h, label=data['strategy'], linewidth=1.5)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Epsilon')
        axes[1].set_title(f'Exploration Rate Over Time - {inst_name}', fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'epsilon_comparison_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: epsilon_comparison_{inst_name}.png")


def plot_best_tour(all_results):
    """Plot the best tour found by Q-Learning on each instance."""
    for inst_name in ALL_INSTANCES:
        strategies = all_results.get(inst_name, {})
        ql_data = strategies.get('Q-Learning (Adaptive)')
        if ql_data is None:
            continue

        coords, _, optimal = get_instance(inst_name)
        tour = ql_data['best_result']['best_tour']
        length = ql_data['best_result']['best_length']

        fig, ax = plt.subplots(figsize=(8, 8))
        coords_arr = np.array(coords)

        # Plot edges
        for i in range(len(tour)):
            c1 = coords_arr[tour[i]]
            c2 = coords_arr[tour[(i + 1) % len(tour)]]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'b-', linewidth=0.8, alpha=0.6)

        # Plot cities
        ax.scatter(coords_arr[:, 0], coords_arr[:, 1], c='red', s=30, zorder=5)

        ax.set_title(f'Best Tour — {inst_name}\n'
                     f'Length: {length:.0f}  (Optimal: {optimal}, Gap: {(length-optimal)/optimal*100:.2f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'tour_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: tour_{inst_name}.png")


def plot_summary_table(all_results):
    """Create a summary comparison table as an image."""
    fig, ax = plt.subplots(figsize=(14, len(ALL_INSTANCES) * 2.5 + 1))
    ax.axis('off')

    # Build table data
    strategies_list = ['Q-Learning (Adaptive)', 'Fixed 2-opt', 'Fixed swap',
                       'Fixed relocate', 'Fixed restart', 'Random']
    col_labels = ['Instance'] + strategies_list
    table_data = []

    for inst_name in ALL_INSTANCES:
        row = [f"{inst_name}\n(opt={OPTIMAL[inst_name]})"]
        for sname in strategies_list:
            data = all_results[inst_name].get(sname, {})
            if data:
                row.append(f"{data['avg_gap']:.2f}%\n±{data['std_gap']:.2f}")
            else:
                row.append('N/A')
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Color the header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight Q-Learning column
    for i in range(len(table_data)):
        table[i + 1, 1].set_facecolor('#E2EFDA')

    ax.set_title('Summary: Average Gap to Optimal (%)', fontsize=14,
                 fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_table.png")


def plot_operator_distribution(all_results):
    """Plot operator usage distribution for Q-Learning."""
    for inst_name in ALL_INSTANCES:
        ql_data = all_results[inst_name].get('Q-Learning (Adaptive)')
        if ql_data is None or ql_data.get('best_result') is None:
            continue

        op_hist = ql_data['best_result']['operator_history']
        counts = [op_hist.count(i) for i in range(NUM_OPERATORS)]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
        bars = ax.bar(OPERATOR_NAMES, counts, color=colors, edgecolor='black', linewidth=0.5)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Operator', fontsize=12)
        ax.set_ylabel('Times Selected', fontsize=12)
        ax.set_title(f'Operator Selection Distribution — {inst_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'operator_dist_{inst_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: operator_dist_{inst_name}.png")


# ============================================================
# Generalization Test
# ============================================================

def run_generalization_test(all_results):
    """Train on small instances, test on larger ones."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Generalization Test (Train on small -> Test on large)")
    print("=" * 70)

    # Train a Q-table on eil51
    _, dist_train, opt_train = get_instance('eil51')
    agent = QLearningAgent(
        alpha=0.1, gamma=0.9,
        epsilon_strategy=AdaptiveEpsilon(),
        reward_fn=reward_pure_improvement,
    )
    # Train over multiple episodes
    for ep in range(10):
        agent.run_episode(dist_train, opt_train, max_iterations=MAX_ITERATIONS,
                          seed=SEED_BASE + ep)
    trained_q_table = agent.q_table.copy()
    print(f"  Trained Q-table on eil51 ({10} episodes)")

    # Test on larger instances with the trained Q-table
    for test_name in TEST_INSTANCES:
        _, dist_test, opt_test = get_instance(test_name)
        gaps_pretrained = []
        gaps_fresh = []

        for run in range(NUM_RUNS):
            # Pre-trained agent
            agent_pre = QLearningAgent(
                alpha=0.05, gamma=0.9,
                epsilon_strategy=ConstantEpsilon(0.05),
                reward_fn=reward_pure_improvement,
            )
            agent_pre.q_table = trained_q_table.copy()
            result_pre = agent_pre.run_episode(dist_test, opt_test,
                                                max_iterations=MAX_ITERATIONS,
                                                seed=SEED_BASE + run)
            gaps_pretrained.append(result_pre['gap'])

            # Fresh agent
            agent_fresh = QLearningAgent(
                alpha=0.1, gamma=0.9,
                epsilon_strategy=AdaptiveEpsilon(),
                reward_fn=reward_pure_improvement,
            )
            result_fresh = agent_fresh.run_episode(dist_test, opt_test,
                                                    max_iterations=MAX_ITERATIONS,
                                                    seed=SEED_BASE + run)
            gaps_fresh.append(result_fresh['gap'])

        print(f"\n  {test_name}:")
        print(f"    Pre-trained Q-table: {np.mean(gaps_pretrained):.2f}% ± {np.std(gaps_pretrained):.2f}")
        print(f"    Fresh Q-Learning:    {np.mean(gaps_fresh):.2f}% ± {np.std(gaps_fresh):.2f}")


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    start_time = time.time()

    print("Adaptive Local Search Operator Selection for TSP via Q-Learning")
    print("=" * 70)
    print(f"Max iterations per episode: {MAX_ITERATIONS}")
    print(f"Number of runs per strategy: {NUM_RUNS}")
    print(f"Training instances: {TRAIN_INSTANCES}")
    print(f"Test instances: {TEST_INSTANCES}")

    # Run experiments
    all_results = run_main_comparison()
    reward_results = run_reward_comparison()
    epsilon_results = run_epsilon_comparison()
    run_generalization_test(all_results)

    # Generate all visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_convergence(all_results)
    plot_transition_heatmap(all_results)
    plot_reward_comparison(reward_results)
    plot_epsilon_comparison(epsilon_results)
    plot_best_tour(all_results)
    plot_operator_distribution(all_results)
    plot_summary_table(all_results)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE — Total time: {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")

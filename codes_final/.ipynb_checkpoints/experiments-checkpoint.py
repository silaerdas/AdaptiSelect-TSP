""" AdaptiSelect-TSP Experiment Driver """
from __future__ import annotations
import pickle, time
import numpy as np
from scipy.stats import wilcoxon

from tsp_data import get_instance, OPTIMAL
from tsp_operators import OPERATOR_NAMES, NUM_OPERATORS
from tsp_qlearning import (
    QLearningAgent, ConstantEpsilon, ExponentialDecayEpsilon, AdaptiveEpsilon,
    reward_pure_improvement, reward_global_bonus,
    run_fixed_operator, run_random_operator,
)

# Configuration Parameters
NUM_RUNS = 30
MAX_ITERATIONS = 3000
SEED_BASE = 42
TRAIN_INSTANCES = ["eil51", "berlin52"]
TEST_INSTANCES = ["st70", "kroA100"]
ALL_INSTANCES = TRAIN_INSTANCES + TEST_INSTANCES
BASELINES = ["Fixed 2-opt", "Fixed swap", "Fixed relocate", "Fixed restart", "Random"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def main():
    R = {"meta": {"NUM_RUNS": NUM_RUNS, "MAX_ITERATIONS": MAX_ITERATIONS,
                  "SEED_BASE": SEED_BASE, "OPERATOR_NAMES": OPERATOR_NAMES,
                  "OPTIMAL": dict(OPTIMAL), "TRAIN": TRAIN_INSTANCES,
                  "TEST": TEST_INSTANCES}}

    # Experiment 1: Main Baselines Comparison
    log("Experiment 1: main comparison (Q-learning vs fixed operators vs random)")
    main_res = {}
    for inst in ALL_INSTANCES:
        _, D, opt = get_instance(inst)
        rec = {"gaps": {}, "hist": {}}
        ql_gaps, ql_hist = [], []
        agg_trans = np.zeros((NUM_OPERATORS, NUM_OPERATORS))
        agg_improve = np.zeros(NUM_OPERATORS)
        best_gap, best_tour, best_len = float("inf"), None, None
        
        for r in range(NUM_RUNS):
            agent = QLearningAgent(alpha=0.1, gamma=0.9,
                                   epsilon_strategy=AdaptiveEpsilon(),
                                   reward_fn=reward_pure_improvement)
            res = agent.run_episode(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)
            ql_gaps.append(res["gap"]); ql_hist.append(res["length_history"])
            agg_trans += res["transition_counts"]; agg_improve += res["improve_counts"]
            if res["gap"] < best_gap:
                best_gap, best_tour, best_len = res["gap"], res["best_tour"], res["best_length"]
        rec["gaps"]["Q-Learning (Adaptive)"] = np.array(ql_gaps)
        rec["hist"]["Q-Learning (Adaptive)"] = np.array(ql_hist).mean(axis=0)
        rec["agg_transitions"] = agg_trans
        rec["agg_improvements"] = agg_improve
        rec["best_tour"] = best_tour
        rec["best_len"] = best_len

        # Run fixed operators
        for b in BASELINES[:-1]:
            op_idx = OPERATOR_NAMES.index(b.split()[1])
            bg, bh = [], []
            for r in range(NUM_RUNS):
                res = run_fixed_operator(D, opt, op_idx, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)
                bg.append(res["gap"]); bh.append(res["length_history"])
            rec["gaps"][b] = np.array(bg)
            rec["hist"][b] = np.array(bh).mean(axis=0)

        # Run random operator selection
        rg, rh = [], []
        for r in range(NUM_RUNS):
            res = run_random_operator(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)
            rg.append(res["gap"]); rh.append(res["length_history"])
        rec["gaps"]["Random"] = np.array(rg)
        rec["hist"]["Random"] = np.array(rh).mean(axis=0)
        main_res[inst] = rec
        log(f"  {inst} done. QL best gap: {best_gap:.2f}%")
    R["main"] = main_res

    # Experiment 2: Reward Function Formulations
    log("Experiment 2: Reward function comparison (R1 vs R2)")
    rew_res = {}
    for inst in TRAIN_INSTANCES:
        _, D, opt = get_instance(inst)
        g1, g2 = [], []
        for r in range(NUM_RUNS):
            a1 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=AdaptiveEpsilon(), reward_fn=reward_pure_improvement)
            g1.append(a1.run_episode(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)["gap"])
            a2 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=AdaptiveEpsilon(), reward_fn=reward_global_bonus)
            g2.append(a2.run_episode(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)["gap"])
        rew_res[inst] = {"R1": np.array(g1), "R2": np.array(g2)}
    R["reward"] = rew_res

    # Experiment 3: Exploration Strategies
    log("Experiment 3: Exploration strategy comparison")
    eps_res = {}
    for inst in TRAIN_INSTANCES:
        _, D, opt = get_instance(inst)
        strats = {
            "Constant": lambda: ConstantEpsilon(0.1),
            "Decay": lambda: ExponentialDecayEpsilon(1.0, 0.01, 0.995),
            "Adaptive": lambda: AdaptiveEpsilon()
        }
        inst_e = {}
        for name, s_factory in strats.items():
            gaps, ehist = [], []
            for r in range(NUM_RUNS):
                agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=s_factory(), reward_fn=reward_pure_improvement)
                res = agent.run_episode(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)
                gaps.append(res["gap"])
                if r == 0: ehist = res["epsilon_history"]
            inst_e[name] = {"gaps": np.array(gaps), "ehist": np.array(ehist)}
        eps_res[inst] = inst_e
    R["epsilon"] = eps_res

    # Experiment 4: Generalisation Success and Transfer Learning
    log("Experiment 4: Generalisation and Transfer learning")
    combined_q = np.zeros((120, NUM_OPERATORS))
    for inst in TRAIN_INSTANCES:
        _, D, opt = get_instance(inst)
        for r in range(NUM_RUNS):
            agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=AdaptiveEpsilon(), reward_fn=reward_pure_improvement)
            agent.run_episode(D, opt, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)
            combined_q += agent.q_table
    combined_q /= (len(TRAIN_INSTANCES) * NUM_RUNS)

    gen = {}
    for test in TEST_INSTANCES:
        _, Dte, opte = get_instance(test)
        fresh, warm = [], []
        for r in range(NUM_RUNS):
            af = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=AdaptiveEpsilon(), reward_fn=reward_pure_improvement)
            fresh.append(af.run_episode(Dte, opte, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)["gap"])
            aw = QLearningAgent(alpha=0.1, gamma=0.9, epsilon_strategy=AdaptiveEpsilon(), reward_fn=reward_pure_improvement)
            aw.q_table = combined_q.copy()
            warm.append(aw.run_episode(Dte, opte, max_iterations=MAX_ITERATIONS, seed=SEED_BASE + r)["gap"])
        gen[test] = {"fresh": np.array(fresh), "warm": np.array(warm)}
        log(f"  {test}: fresh {np.mean(fresh):.2f} | warm {np.mean(warm):.2f}")
    R["gen"] = gen
    R["combined_q"] = combined_q

    # Statistical Significance Testing (Wilcoxon)
    gsig = {}
    for test in TEST_INSTANCES:
        fr = gen[test]["fresh"]; wm = gen[test]["warm"]
        if np.allclose(fr, wm):
            gsig[test] = {"p": 1.0, "mean_diff": 0.0}
        else:
            _, p = wilcoxon(fr, wm)
            gsig[test] = {"p": float(p), "mean_diff": float((warm - fr).mean())}
    R["gen_sig"] = gsig

    # Save Output
    with open("results.pkl", "wb") as f:
        pickle.dump(R, f, protocol=4)
    log("All experiment results saved to results.pkl.")

if __name__ == "__main__":
    main()
# AdaptiSelect-TSP: Adaptive Local Search Operator Selection for TSP via Tabular Q-Learning

**AIE 635 – Reinforcement Learning | Hacettepe University**  
**Authors:** Sıla Erdaş, Özlem Akyaz

---

## Project Overview

This project implements a **tabular Q-learning based selection hyper-heuristic** for the **Traveling Salesperson Problem (TSP)**. The main goal is to improve local search by selecting among different operators during the search process instead of using only one fixed operator.

The agent chooses between four local search operators:

- 2-opt
- Swap
- Relocate
- Restart

The search process is modeled as a **120-state Markov Decision Process (MDP)**. The state representation includes solution quality, whether the last step improved the solution, the search stage, and the last operator used.

The project is evaluated on four TSPLIB benchmark instances:

- `eil51`
- `berlin52`
- `st70`
- `kroA100`

---

## Key Features

- Tabular Q-learning for adaptive operator selection
- 120-state MDP formulation
- Four local search operators: 2-opt, swap, relocate, restart
- Three epsilon-greedy exploration strategies:
  - Constant epsilon
  - Exponential decay
  - Adaptive stagnation-based epsilon
- Two reward definitions:
  - R1: pure improvement reward
  - R2: global-best bonus reward
- Operator transition analysis to observe useful operator sequences
- Generalization test on larger unseen instances

---

## Repository Structure

```text
AdaptiSelect-TSP/
├── src/
│   ├── tsp_data.py          # TSPLIB instance data and distance matrix
│   ├── tsp_operators.py     # Local search operators
│   ├── tsp_qlearning.py     # Q-learning agent, reward functions, epsilon strategies
│   └── main.py              # Main experiment runner
├── report/
│   └── Final_Report_Adaptive_TSP_Q_Learning.docx
│   └── main_final.tex
├── results/                 # Generated figures and output files
├── README.md
└── requirements.txt
```

---

## Requirements

The project requires Python 3.8+ and the following libraries:

- numpy
- matplotlib
- seaborn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AdaptiSelect-TSP.git
cd AdaptiSelect-TSP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the experiments

```bash
python src/main.py
```

After running the code, the generated figures and output files will be saved in the `results/` folder.

---

## Experiments

The project includes four main experimental parts:

| Experiment | Description |
|---|---|
| Experiment 1 | Q-learning vs. fixed-operator and random baselines |
| Experiment 2 | Reward shaping comparison: R1 vs. R2 |
| Experiment 3 | Epsilon strategy comparison: constant, exponential decay, adaptive stagnation |
| Experiment 4 | Generalization test across different TSP instance sizes |

---

## Generated Outputs

The `results/` folder may include the following outputs:

- `convergence_<instance>.png` — convergence curves for each TSP instance
- `transition_heatmap_<instance>.png` — operator transition heatmaps
- `reward_comparison.png` — reward shaping comparison
- `epsilon_strategy_comparison.png` — epsilon strategy comparison
- `restart_count.png` — average restart counts
- `qtable_heatmap_<instance>.png` — Q-table analysis by quality band
- `best_tour_<instance>.png` — best tour visualization
- `generalization_comparison.png` — pre-trained vs. fresh Q-learning results

File names may vary slightly depending on the script version.

---

## Results Summary

Average optimality gap (%) over 5 runs:

| Strategy | eil51 | berlin52 | st70 | kroA100 |
|---|---:|---:|---:|---:|
| Q-Learning (Adaptive) | **2.91 ± 1.44** | **0.60 ± 1.72** | **3.20 ± 0.92** | 6.06 ± 2.45 |
| Fixed 2-opt | 3.94 ± 1.55 | 2.17 ± 2.41 | 5.42 ± 1.80 | **5.10 ± 0.62** |
| Fixed swap | 22.35 ± 3.19 | 17.12 ± 4.17 | 27.70 ± 5.31 | 27.72 ± 3.59 |
| Fixed relocate | 19.62 ± 2.55 | 17.20 ± 4.10 | 27.08 ± 5.80 | 27.51 ± 4.88 |
| Fixed restart | 24.04 ± 4.25 | 17.50 ± 4.30 | 27.97 ± 5.51 | 28.27 ± 4.07 |
| Random selection | 4.23 ± 1.48 | 1.82 ± 2.26 | 5.36 ± 2.01 | 12.30 ± 3.10 |

The Q-learning approach performs better than the fixed-operator baselines on 3 out of 4 instances. On `kroA100`, Fixed 2-opt gives a slightly lower mean gap than the Q-learning agent.

---

## Main Observations

- Adaptive stagnation-based epsilon gives the best overall results among the tested exploration strategies.
- R1 and R2 reward definitions give very similar results in the experiments.
- Operator transition analysis shows that useful operator sequences can change depending on the problem instance.
- The Q-table suggests that the agent changes its operator choices depending on the quality of the current solution.
- Direct transfer of a learned Q-table to larger instances does not improve performance, suggesting that better state normalization may be needed.

---

## MDP Formulation

| Component | Description |
|---|---|
| States | 120 states: 5 quality bands × 2 improvement flags × 3 search stages × 4 last operators |
| Actions | {2-opt, swap, relocate, restart} |
| Reward R1 | Previous tour length − new tour length |
| Reward R2 | R1 + bonus if a new global best solution is found |
| Policy | Epsilon-greedy action selection |
| Learning method | Tabular Q-learning |

---

## Local Search Operators

| Operator | Description |
|---|---|
| 2-opt | Reverses a tour segment to improve the route |
| Swap | Exchanges the positions of two cities |
| Relocate | Removes one city and inserts it into another position |
| Restart | Generates a new random tour |

---

## Report

The final project report is included in the `report/` folder.
The .tex file is also included in this folder.

---

## References

1. Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. Freeman.
2. Applegate, D., Bixby, R., Chvatal, V., & Cook, W. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.
3. Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan traveling salesman heuristic. *European Journal of Operational Research*.
4. Burke, E. K. et al. (2010). A classification of hyper-heuristic approaches. In *Handbook of Metaheuristics*. Springer.
5. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*.
6. Lin, S. (1965). Computer solutions of the traveling salesman problem. *Bell System Technical Journal*.
7. Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem. *Operations Research*.
8. Or, I. (1976). *Traveling Salesman-type Combinatorial Problems and Their Relation to the Logistics of Regional Blood Banking*. Ph.D. Dissertation.
9. Bello, I. et al. (2016). Neural combinatorial optimization with reinforcement learning. arXiv:1611.09940.
10. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, learn to solve routing problems! *ICLR*.

---

## Notes

This repository was prepared for the AIE 635 Reinforcement Learning course project.

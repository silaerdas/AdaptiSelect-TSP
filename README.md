# AdaptiSelect-TSP: Phase-Aware Q-Learning for Adaptive Local Search Operator Selection

**AIE 635 – Reinforcement Learning | Hacettepe University**  
**Authors:** Sıla Erdaş, Özlem Akyaz

---

## Project Overview

This project implements a **Tabular Q-Learning hyper-heuristic** that adaptively selects among four local search operators (2-opt, Swap, Relocate, Restart) to solve the **Travelling Salesman Problem (TSP)**. The search process is modeled as a **120-state Markov Decision Process (MDP)** encoding solution quality, iteration stage, recent improvement, and last operator used.

### Key Features
- Phase-aware adaptive operator selection via Q-Learning
- Two reward functions: R1 (pure improvement) vs R2 (global-best bonus)
- Adaptive epsilon-greedy with stagnation detection
- Operator transition matrix for synergy analysis
- Evaluated on TSPLIB benchmark instances: eil51, berlin52, st70, kroA100

---

## Repository Structure

```
AdaptiSelect-TSP/
├── src/
│   ├── tsp_data.py          # TSPLIB instance coordinates and distance matrix
│   ├── tsp_operators.py     # Local search operators (2-opt, swap, relocate, restart)
│   ├── tsp_qlearning.py     # Q-Learning agent, reward functions, epsilon strategies
│   └── main.py              # Main runner: experiments and visualizations
├── report/
│   └── AIE635_FinalReport.docx   # Project report
├── results/                 # Generated figures and output files (auto-created)
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- seaborn

Install all dependencies:

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

### 3. Run all experiments

```bash
python src/main.py
```

This will run all four experiments and save results to the `results/` folder:

| Experiment | Description |
|---|---|
| Experiment 1 | Q-Learning vs baseline strategies (Fixed 2-opt, Swap, Relocate, Restart, Random) |
| Experiment 2 | Reward shaping comparison (R1 vs R2) |
| Experiment 3 | Epsilon strategy comparison (Constant, Exponential Decay, Adaptive) |
| Experiment 4 | Generalization test (train on eil51, test on st70 and kroA100) |

### 4. Generated outputs

After running, the `results/` folder will contain:

- `convergence_<instance>.png` — convergence curves per instance
- `transition_heatmap_<instance>.png` — operator synergy heatmaps
- `reward_comparison_<instance>.png` — R1 vs R2 convergence
- `epsilon_comparison_<instance>.png` — epsilon strategy plots
- `tour_<instance>.png` — best tour visualization
- `operator_dist_<instance>.png` — operator usage distribution
- `summary_table.png` — full results table

---

## Results Summary

Average optimality gap (%) across 5 seeds:

| Instance | Q-Learning (Adaptive) | Fixed 2-opt | Random |
|---|---|---|---|
| eil51 (opt=426) | **3.00 ±1.07** | 3.94 ±1.55 | 3.99 ±0.71 |
| berlin52 (opt=7542) | **1.03 ±1.01** | 2.17 ±2.41 | 1.28 ±2.67 |
| st70 (opt=675) | **3.82 ±2.06** | 5.42 ±1.80 | 6.99 ±2.69 |
| kroA100 (opt=21282) | **3.90 ±0.76** | 5.10 ±0.62 | 16.68 ±3.13 |

---

## MDP Formulation

| Component | Description |
|---|---|
| **States** | 120 = 5 (quality bands) × 2 (improved flag) × 3 (iteration stage) × 4 (last operator) |
| **Actions** | {2-opt, Swap, Relocate, Restart} |
| **Reward R1** | prev_length − new_length |
| **Reward R2** | R1 + bonus if new global best found |
| **Policy** | ε-greedy with adaptive stagnation detection |

---

## References

1. Watkins & Dayan (1992). Q-Learning. Machine Learning.
2. Mazyavkina et al. (2021). RL for Combinatorial Optimization: A Survey. Computers & OR.
3. Chung et al. (2025). Neural Combinatorial Optimization with RL. AI Review.
4. Khalil et al. (2017). Learning Combinatorial Optimization over Graphs. NeurIPS.
5. Burke et al. (2013). Hyper-Heuristics: A Survey. JORS.
6. Karimi-Mamaghan et al. (2021). Learning-Based ILS for TSP. OLA 2021.
7. Kool et al. (2019). Attention, Learn to Solve Routing Problems. ICLR.
8. Reinelt (1991). TSPLIB. ORSA Journal on Computing.

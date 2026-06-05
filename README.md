# AdaptiSelect-TSP

**AIE635 Reinforcement Learning Final Project**  
**Adaptive Operator Selection for the Travelling Salesman Problem via Tabular Q-Learning**  
**Authors:** Özlem Akyaz, Sıla Erdaş

---

## Project Overview

This repository contains our AIE635 final project on adaptive local-search operator selection for the symmetric Travelling Salesman Problem (TSP).

The project models an iterated local-search process as a Markov Decision Process (MDP). A tabular Q-learning agent learns to choose among four operators:

- 2-opt
- swap
- relocate
- random restart

The goal is to test whether a learned, state-aware operator selection policy can perform better than using a single fixed operator throughout the search.

---

## Repository Structure

```text
AdaptiSelect-TSP/
├── codes_final/      # Final version of the Python implementation
├── codes_v1/         # Earlier version of the code
├── latex_final/      # Final LaTeX report files
├── report_v1/        # Previous report version
├── results_v1/       # Generated result figures and outputs
├── aie635_finalReport.pdf
├── README.md
└── requirements.txt
```

---

## How to Run the Final Code

1. Open the project folder:

```bash
cd AdaptiSelect-TSP
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Go to the final code folder:

```bash
cd codes_final
```

4. Run the main Python file. Depending on the final file name in the folder, run for example:

```bash
python main.py
```

or:

```bash
python main1.py
```

The generated plots and outputs can be saved into the results folder.

---

## Experiments

The final report includes the following main experiments:

- Q-learning operator selection compared with fixed 2-opt, fixed swap, fixed relocate, fixed restart, and random selection
- Reward comparison between raw improvement reward and global-best bonus reward
- Exploration schedule comparison:
  - constant epsilon
  - exponential decay
  - adaptive stagnation-based epsilon
- Sensitivity analysis of the stagnation signal
- Transfer test from smaller instances to larger instances
- Operator-transition analysis

The experiments are performed on four TSPLIB instances:

- eil51
- berlin52
- st70
- kroA100

---

## Main Results

The Q-learning policy achieves the best mean gap on three instances:

| Instance | Q-Learning Gap |
|---|---:|
| eil51 | 3.90% |
| berlin52 | 0.51% |
| st70 | 3.34% |
| kroA100 | 6.23% |

For `kroA100`, fixed 2-opt performs better than Q-learning. This shows that adaptive operator selection is useful in some cases, but its advantage decreases on the largest tested instance.

The adaptive stagnation-based epsilon strategy gives the best overall exploration performance. However, the results also show that the method is sensitive to how stagnation is defined.

---

## Report

The final report is provided as:

```text
aie635_finalReport.pdf
```

The LaTeX source files are available in:

```text
latex_final/
```

Older report files are kept in:

```text
report_v1/
```

---

## Notes on Versions

- `codes_final/` contains the final implementation used for the final report.
- `codes_v1/` contains an earlier code version.
- `latex_final/` contains the final report source.
- `report_v1/` and `results_v1/` are kept for reference.

For grading or review, please use the final code and final report files.

---

## Requirements

The project uses Python and common scientific libraries. The required packages are listed in:

```text
requirements.txt
```

Typical dependencies include:

- numpy
- matplotlib
- seaborn

---

## References

The detailed references are included in the final report.

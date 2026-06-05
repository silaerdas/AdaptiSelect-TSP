# AdaptiSelect-TSP — Q-Learning for Adaptive Operator Selection on the TSP

Tabular Q-learning that selects, at each step of a local search, which of four
operators to apply to a TSP tour: best-improvement **2-opt**, **swap**,
**relocate**, or random **restart**. The search is modelled as a 120-state MDP
(quality band × improvement flag × stage × last operator).

## Files

| File | Purpose |
|------|---------|
| `tsp_data.py` | TSPLIB instances (eil51, berlin52, st70, kroA100), EUC_2D distance matrix, nearest-neighbour construction. |
| `tsp_operators.py` | The four operators and the operator registry. |
| `tsp_qlearning.py` | Q-learning agent, reward functions (R1/R2), three ε strategies, and the fixed/random baselines. |
| `experiments.py` | Runs all four experiments and writes `results.pkl` (with paired Wilcoxon tests). |
| `make_figures.py` | Reads `results.pkl` and writes every figure into `figures/`. |
| `results.pkl` | Pre-computed results from a full 30-run execution (so figures can be regenerated without re-running). |
| `figures/` | The 14 figures used in the report. |

## How to run

```bash
pip install numpy scipy matplotlib seaborn      # dependencies
python experiments.py                           # ~10 min on one CPU core -> results.pkl
python make_figures.py                          # results.pkl -> figures/*.png
```

`results.pkl` is already included, so you can run `make_figures.py` directly to
reproduce the figures without waiting for the experiments.

### Configuration
At the top of `experiments.py`:
`NUM_RUNS = 30`, `MAX_ITERATIONS = 3000`, `SEED_BASE = 42`, training instances
`eil51, berlin52`, test instances `st70, kroA100`. The same seed set is reused
across strategies so that comparisons use a **paired** Wilcoxon signed-rank test.

## What was corrected relative to the original submission

These are the substantive changes made during the review, with the reasoning:

1. **Run count.** The original driver ran `NUM_RUNS = 10` while the report claimed
   30 runs and "statistical significance". Everything here is run at a real 30
   seeds, and significance is now tested explicitly with a paired Wilcoxon
   signed-rank test (none was performed originally).

2. **Main result is now stated honestly.** Across 30 runs the learned policy
   beats the strongest fixed operator (2-opt) on eil51, berlin52 and st70
   (p < 0.05) but is **worse** than a well-tuned fixed 2-opt on the largest
   instance, kroA100 (5.21 % vs 6.23 %, p = 0.031). The original "comprehensive
   dominance" claim was not supported; the effect is size-dependent.

3. **The "operator synergy" was an artefact.** The reported swap→2-opt
   transition probability of 1.00 is not a learned pairing. Transition counts are
   now **aggregated over all 30 runs** (the original used a single best run) and
   compared against the unconditional marginal: 2-opt is >99.7 % of all improving
   moves, so every operator is followed by 2-opt at the base rate. The swap row
   has only ~70 events, which trivially normalises to 1.00.

4. **Generalisation comparison was confounded.** The original warm-start agent
   used a different learning rate (0.05) and a low constant ε (0.05) than the
   fresh agent (0.1, adaptive ε), so "negative transfer" mixed transfer with a
   weaker learner. The test now uses **matched hyper-parameters**; under matched
   settings transfer is statistically neutral (st70 p = 0.41, kroA100 p = 0.65).

5. **Reward functions.** R1 and R2 produce *identical* gap distributions on both
   training instances under shared seeds (the global-best bonus never flips a
   greedy choice within the budget). This is reported as such, with the mechanism
   explained, rather than left as an unexplained coincidence.

6. **Code cleanup.** Removed an unused plain `two_opt` function (the registry uses
   `two_opt_best`) and an unused `OPERATOR_NAMES` import. The earlier correctness
   fixes are retained: TSPLIB-correct rounding `floor(d+0.5)`, delta-based 2-opt
   evaluation, a `relocate` that always moves the city, guards against degenerate
   inputs, and per-run seeding for reproducibility.

7. **References.** Two bibliography entries were wrong and have been fixed:
   the Lin–Kernighan–Helsgaun reference (single author Keld Helsgaun, EJOR
   126(1):106–130, 2000), and the reinforcement-learning-for-operator-selection
   reference (Meignan, Koukam & Créput, *Journal of Heuristics* 16:859–879, 2010).

## Known limitation

The `quality_band` state feature is computed from the **known optimal tour
length**, which is an oracle unavailable in real deployment. It is retained to
study the selection mechanism, but it is the main threat to external validity and
is the direct reason transfer is neutral (the state distribution is defined
relative to each instance's optimum). See the report's Discussion.

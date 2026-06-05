"""figures"""
from __future__ import annotations
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tsp_data import get_instance

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

with open("results.pkl", "rb") as f:
    R = pickle.load(f)

OPT = R["meta"]["OPTIMAL"]
OPNAMES = R["meta"]["OPERATOR_NAMES"]
ALL = R["meta"]["TRAIN"] + R["meta"]["TEST"]
TRAIN = R["meta"]["TRAIN"]
TEST = R["meta"]["TEST"]

ORDER = ["Q-Learning (Adaptive)", "Fixed 2-opt", "Fixed swap",
         "Fixed relocate", "Fixed restart", "Random"]
NICE = {"Q-Learning (Adaptive)": "Q-Learning", "Fixed 2-opt": "Fixed 2-opt",
        "Fixed swap": "Fixed swap", "Fixed relocate": "Fixed relocate",
        "Fixed restart": "Fixed restart", "Random": "Random"}
PALETTE = {"Q-Learning (Adaptive)": "#C0392B", "Fixed 2-opt": "#2471A3",
           "Fixed swap": "#7D8B99", "Fixed relocate": "#AAB7B8",
           "Fixed restart": "#CACFD2", "Random": "#27AE60"}


def savefig(name):
    plt.tight_layout()
    p = os.path.join(FIGDIR, name)
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("saved", p)


# convergence curves, one per instance 
for inst in ALL:
    hist = R["main"][inst]["hist"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for s in ORDER:
        lw = 2.2 if s == "Q-Learning (Adaptive)" else 1.3
        ax.plot(hist[s], label=NICE[s], color=PALETTE[s], linewidth=lw)
    ax.axhline(OPT[inst], color="black", ls="--", lw=1.0, label=f"Optimal ({OPT[inst]})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best tour length")
    ax.set_title(f"Convergence on {inst}")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    savefig(f"convergence_{inst}.png")


# Transition analysis with marginal comparison
for inst in ["eil51", "kroA100"]:
    C = R["main"][inst]["agg_transitions"].astype(float)
    rs = C.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    P = C / rs
    marg = C.sum(axis=0) / max(C.sum(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.heatmap(C.astype(int), annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=OPNAMES, yticklabels=OPNAMES, ax=axes[0], cbar=False)
    axes[0].set_title(f"Aggregated operator transition counts — {inst}\n(improving move $i$ → next improving move $j$)")
    axes[0].set_xlabel("Next operator $j$"); axes[0].set_ylabel("Previous operator $i$")

    sns.heatmap(P, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=OPNAMES, yticklabels=OPNAMES, ax=axes[1])
    axes[1].set_title(f"Row-normalised P(next=$j$ | prev=$i$) — {inst}")
    axes[1].set_xlabel("Next operator $j$"); axes[1].set_ylabel("Previous operator $i$")
    # annotate marginal under the figure
    mtxt = "  ".join(f"{n}:{m:.3f}" for n, m in zip(OPNAMES, marg))
    fig.text(0.5, -0.02, f"Unconditional marginal P(next=$j$):   {mtxt}",
             ha="center", fontsize=8.5)
    savefig(f"transition_{inst}.png")


# q-table heatmaps: mean Q per quality band x operator 
QB_LABELS = ["<5%\n(excellent)", "5–15%\n(good)", "15–30%\n(fair)",
             "30–50%\n(poor)", ">50%\n(very poor)"]
for inst in TRAIN:
    Q = R["qtab"][inst]  # (120, 4)
    M = np.zeros((5, len(OPNAMES)))
    for qb in range(5):
        rows = [s for s in range(Q.shape[0]) if s // 24 == qb]
        M[qb] = Q[rows].mean(axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    sns.heatmap(M, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=OPNAMES, yticklabels=QB_LABELS, ax=ax)
    ax.set_title(f"Mean learned Q-value by quality band — {inst}")
    ax.set_xlabel("Operator"); ax.set_ylabel("Solution-quality band")
    savefig(f"qtable_{inst}.png")


# reward shaping R1 vs R2 (convergence, both training instances) 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, inst in zip(axes, TRAIN):
    for rname, color in [("R1 (Pure)", "#2471A3"), ("R2 (Global Bonus)", "#C0392B")]:
        d = R["reward"][f"{inst}_{rname}"]
        ls = "-" if rname.startswith("R1") else "--"
        ax.plot(d["hist"], label=rname, color=color, ls=ls, lw=1.8)
    ax.axhline(OPT[inst], color="black", ls=":", lw=1.0, label=f"Optimal ({OPT[inst]})")
    ax.set_title(f"Reward shaping — {inst}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best tour length")
    ax.legend(fontsize=8)
savefig("reward_comparison.png")


# epsilon strategies: (eil51) 
inst = "eil51"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
estrats = ["Constant eps=0.1", "Exponential Decay", "Adaptive (Stagnation)"]
ecol = {"Constant eps=0.1": "#7D8B99", "Exponential Decay": "#E67E22",
        "Adaptive (Stagnation)": "#C0392B"}
for ename in estrats:
    d = R["eps"][f"{inst}_{ename}"]
    axes[0].plot(d["hist"], label=ename, color=ecol[ename], lw=1.7)
axes[0].axhline(OPT[inst], color="black", ls=":", lw=1.0, label="Optimal")
axes[0].set_title(f"Convergence by exploration strategy — {inst}")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Best tour length")
axes[0].legend(fontsize=8)
for ename in estrats:
    d = R["eps"][f"{inst}_{ename}"]
    eh = np.asarray(d["eps_history"], dtype=float)
    if eh.size:
        axes[1].plot(eh, label=ename, color=ecol[ename], lw=1.5)
axes[1].set_title(f"Exploration rate $\\epsilon$ over time — {inst}")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel(r"$\epsilon$")
axes[1].legend(fontsize=8)
savefig("epsilon_comparison.png")

# 5b) Restart counts by strategy (both training instances) ------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(estrats)); w = 0.36
for k, inst in enumerate(TRAIN):
    means = [R["eps"][f"{inst}_{e}"]["restarts"].mean() for e in estrats]
    ax.bar(x + (k - 0.5) * w, means, w, label=inst)
ax.set_xticks(x); ax.set_xticklabels(["Constant", "Exp. Decay", "Adaptive"])
ax.set_ylabel("Mean restarts per episode")
ax.set_title("Restart frequency by exploration strategy")
ax.legend()
savefig("epsilon_restarts.png")


# 
fig, ax = plt.subplots(figsize=(6.0, 4.0))
x = np.arange(len(TEST)); w = 0.36
fresh_m = [R["gen"][t]["fresh"].mean() for t in TEST]
fresh_s = [R["gen"][t]["fresh"].std() for t in TEST]
warm_m = [R["gen"][t]["warm"].mean() for t in TEST]
warm_s = [R["gen"][t]["warm"].std() for t in TEST]
ax.bar(x - w/2, fresh_m, w, yerr=fresh_s, capsize=4, label="Fresh (train on test)", color="#2471A3")
ax.bar(x + w/2, warm_m, w, yerr=warm_s, capsize=4, label="Warm-start (transfer)", color="#C0392B")
ax.set_xticks(x); ax.set_xticklabels(TEST)
ax.set_ylabel("Gap to optimal (%)")
ax.set_title("Transfer: warm-started vs fresh agent (matched hyperparameters)")
ax.legend()
savefig("generalization.png")


# Best tour (eil51, kroA100) 
for inst in ["eil51", "kroA100"]:
    coords, _, opt = get_instance(inst)
    coords = np.array(coords, dtype=float)
    tour = R["main"][inst]["best_tour"]
    blen = R["main"][inst]["best_len"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for i in range(len(tour)):
        a = coords[tour[i]]; b = coords[tour[(i + 1) % len(tour)]]
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#2471A3", lw=0.9, alpha=0.7)
    ax.scatter(coords[:, 0], coords[:, 1], c="#C0392B", s=18, zorder=5)
    gap = (blen - opt) / opt * 100
    ax.set_title(f"Best Q-Learning tour — {inst}\nlength {blen:.0f} (opt {opt}, gap {gap:.2f}%)")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    savefig(f"tour_{inst}.png")



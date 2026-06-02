"""Combine the per-model metrics into one table and decompose the effects.

Reads results/<model>/metrics.json produced by the four training notebooks and
writes results/comparison_table.png. Run after the notebooks finish; no GPU needed.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS = os.path.join(BASE, "results")
PLAIN_M1 = 0.2042  # plain-text bert-base from the original project


def load(name):
    with open(os.path.join(RESULTS, name, "metrics.json")) as f:
        return json.load(f)


names = ["M1_marker", "M2", "M3", "M3_LS"]
m = {n: load(n) for n in names}
f1 = {n: m[n]["micro_f1"] for n in names}
ap = {n: m[n]["auprc"] for n in names}

print("model        micro_f1   auprc    best_epoch")
for n in names:
    print(f"{n:11}  {f1[n]:.4f}    {ap[n]:.4f}   {m[n]['best_epoch']}")

marker = f1["M1_marker"] - PLAIN_M1
backbone = f1["M2"] - f1["M1_marker"]
size = f1["M3"] - f1["M2"]
ls = f1["M3_LS"] - f1["M3"]

print()
print(f"marker     (plain bert 0.2042 -> bert+marker):   {marker:+.4f}")
print(f"backbone   (bert-base -> roberta-base):          {backbone:+.4f}")
print(f"size       (roberta-base -> roberta-large):      {size:+.4f}")
print(f"smoothing  (M3 -> M3 + label smoothing):         {ls:+.4f}")

if f1["M1_marker"] >= 0.55 and marker > 0.05:
    print()
    print("the marker carries the gain; the backbone swap adds little on its own")

labels = ["bert\nplain", "bert\n+marker", "roberta-b\n+marker",
          "roberta-l\n+marker", "roberta-l\n+LS"]
f1_vals = [PLAIN_M1, f1["M1_marker"], f1["M2"], f1["M3"], f1["M3_LS"]]
colors = ["#aaaaaa", "#7fbfff", "#4da6ff", "#0066cc", "#004499"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
b = axes[0].bar(labels, f1_vals, color=colors, edgecolor="black", lw=0.7)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("micro f1")
axes[0].set_title("micro f1 (no_relation excluded)")
axes[0].grid(alpha=0.3, axis="y")
for bar, v in zip(b, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", fontsize=9)

au_vals = [ap[n] for n in names]
b2 = axes[1].bar(labels[1:], au_vals, color=colors[1:], edgecolor="black", lw=0.7)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("auprc")
axes[1].set_title("auprc")
axes[1].grid(alpha=0.3, axis="y")
for bar, v in zip(b2, au_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", fontsize=9)

plt.tight_layout()
out = os.path.join(RESULTS, "comparison_table.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

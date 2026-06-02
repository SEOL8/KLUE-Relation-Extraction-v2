"""Inspect the M3 validation predictions saved by m3.ipynb.

Covers three things:
1. The micro-F1 implementation here drops rows whose true label is no_relation,
   which reads slightly higher than the KLUE definition that keeps every row and
   restricts to labels 1..29. Both numbers are printed so the gap is explicit.
2. Per-class F1 to find the weak relations.
3. Where the errors come from: missing a relation, inventing one, or confusing
   two relation types.

Reads m3_val_logits.npy and m3_val_labels.npy. No GPU needed.
"""
import os

import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support

BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

LABELS = [
    "no_relation", "org:dissolved", "org:founded", "org:place_of_headquarters",
    "org:alternate_names", "org:member_of", "org:members",
    "org:political/religious_affiliation", "org:product", "org:founded_by",
    "org:top_members/employees", "org:number_of_employees/members",
    "per:date_of_birth", "per:date_of_death", "per:place_of_birth",
    "per:place_of_death", "per:place_of_residence", "per:origin",
    "per:employee_of", "per:schools_attended", "per:alternate_names",
    "per:parents", "per:children", "per:siblings", "per:spouse",
    "per:other_family", "per:colleagues", "per:product", "per:religion",
    "per:title",
]

y = np.load(os.path.join(BASE, "m3_val_labels.npy"))
p = np.load(os.path.join(BASE, "m3_val_logits.npy")).argmax(1)
print("validation samples:", len(y))

# 1. two micro-F1 conventions
masked = f1_score(y[y != 0], p[y != 0], average="micro", zero_division=0)
official = f1_score(y, p, labels=list(range(1, 30)), average="micro", zero_division=0)
fp_norel = int(((y == 0) & (p != 0)).sum())
print()
print(f"micro f1, true-no_relation rows dropped : {masked:.4f}")
print(f"micro f1, klue convention (labels 1..29) : {official:.4f}")
print(f"gap {masked - official:+.4f}; {fp_norel} no_relation rows predicted as a relation "
      "count against the second number but not the first")

# 2. per-class F1, support at least 20
pr, rc, fc, sup = precision_recall_fscore_support(
    y, p, labels=list(range(30)), zero_division=0)
keep = sorted([i for i in range(30) if sup[i] >= 20 and i != 0], key=lambda i: fc[i])
names = [LABELS[i].split(":")[-1][:16] for i in keep]
vals = [fc[i] for i in keep]
sizes = [sup[i] for i in keep]
bar_colors = ["#cc3333" if v < 0.55 else "#dd9933" if v < 0.7 else "#3377cc" for v in vals]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(len(keep)), vals, color=bar_colors, edgecolor="black", lw=0.5)
ax.set_yticks(range(len(keep)))
ax.set_yticklabels([f"{n} (n={s})" for n, s in zip(names, sizes)], fontsize=8)
ax.set_xlim(0, 1)
ax.set_xlabel("f1 (klue convention)")
ax.set_title("M3 per-class f1, support >= 20")
for bar, v in zip(bars, vals):
    ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=7)
ax.axvline(official, color="gray", ls="--", alpha=0.7, label=f"micro f1 = {official:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "M3_per_class_f1.png"), dpi=130)
plt.close()

# 3. error composition
over = int(((y == 0) & (p != 0)).sum())
miss = int(((y != 0) & (p == 0)).sum())
swap = int(((y != 0) & (p != 0) & (y != p)).sum())
total = over + miss + swap

fig, ax = plt.subplots(figsize=(7, 4))
cats = ["no_relation as relation\n(over-detect)",
        "relation as no_relation\n(miss)",
        "relation as wrong relation\n(type swap)"]
counts = [over, miss, swap]
bars = ax.bar(cats, counts, color=["#cc6666", "#ddaa55", "#6699cc"], edgecolor="black")
ax.set_ylabel("count")
ax.set_title(f"M3 error composition, total {total}")
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 8, f"{c}\n({c / total * 100:.0f}%)",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "M3_error_composition.png"), dpi=130)
plt.close()

print()
print(f"errors total {total}: over-detect {over}, miss {miss}, type-swap {swap}")
print("most errors are about whether a relation exists at all, not which relation it is")
print("wrote results/M3_per_class_f1.png and results/M3_error_composition.png")

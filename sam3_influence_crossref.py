"""
SAM3 influence analysis — cross-referencing standalone vs pipeline predictions.

Two analyses:
  A) SAM3 hurt MedGemma:
     Images where the pipeline predicted WRONG and standalone predicted RIGHT.
     i.e. MedGemma was fine on its own but SAM3's bounding box misled it.

  B) SAM3 helped MedGemma:
     Images where standalone predicted WRONG and the pipeline predicted RIGHT.
     i.e. SAM3's bounding box corrected a MedGemma mistake.

Usage: python sam3_influence_crossref.py
Outputs:
  results/sam3_influence_crossref.txt
  results/plots/11_sam3_hurt_vs_helped.png
  results/plots/12_sam3_hurt_breakdown.png
  results/plots/13_sam3_helped_breakdown.png
"""

import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STANDALONE_PATH = "results/phases/standalone_summary_results.jsonl"
SAM3_PATH       = "results/sam3_summary_results.jsonl"
OUT_TXT         = "results/sam3_influence_crossref.txt"
OUT_PLOTS       = "results/plots"
os.makedirs(OUT_PLOTS, exist_ok=True)

TUMOR_DATASETS  = {"images-17", "images-44c", "figshare", "br35h"}

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GRAY   = "#8C8C8C"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get(rec, path):
    obj = rec
    for k in path.split("."):
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj

def norm_path(p):
    for prefix in ("data/raw data/", "data/"):
        if str(p or "").startswith(prefix):
            return str(p)[len(prefix):]
    return str(p or "")

def pct(c, t):
    return c / t * 100 if t else 0.0

_lines = []
def pr(s=""):
    print(s)
    _lines.append(s)

def save(fig, name):
    p = os.path.join(OUT_PLOTS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")

# ── load data ─────────────────────────────────────────────────────────────────

standalone_all = load_jsonl(STANDALONE_PATH)
standalone = [r for r in standalone_all
              if (get(r, "input.metadata.dataset") or "").lower() in TUMOR_DATASETS]

sam3_all = load_jsonl(SAM3_PATH)
sam3 = [r for r in sam3_all if r.get("mode") == "sam3_bbox"]

# ── build lookup: normalised image path → standalone record ──────────────────

standalone_by_path = {}
for r in standalone:
    ip = norm_path(get(r, "input.image_path") or "")
    standalone_by_path[ip] = r

# ── for each sam3 record, find its standalone counterpart ────────────────────

def standalone_pred(r):
    """diagnosis_name prediction from standalone record for the same image."""
    ip = norm_path(r.get("image_path", ""))
    sr = standalone_by_path.get(ip)
    if sr is None:
        return None
    return (get(sr, "output.parsed_response.diagnosis_name") or "").lower()

def pipeline_pred(r):
    return ((r.get("prediction") or {}).get("diagnosis_name") or "").lower()

def gt_class(r):
    label = r.get("gt_label", "normal")
    return "normal" if label == "normal" else "tumor"

def gt_subclass(r):
    label = r.get("gt_label", "normal")
    return None if label == "normal" else label

# ── Analysis A: SAM3 HURT MedGemma ───────────────────────────────────────────
# pipeline wrong AND standalone right

hurt = []
for r in sam3:
    gt  = gt_class(r)
    pp  = pipeline_pred(r)
    sp  = standalone_pred(r)
    if sp is None:
        continue
    pipeline_wrong    = (pp != gt)
    standalone_right  = (sp == gt)
    if pipeline_wrong and standalone_right:
        hurt.append({
            "image_path":     r.get("image_path"),
            "gt":             gt,
            "gt_sub":         gt_subclass(r),
            "standalone_pred": sp,
            "pipeline_pred":  pp,
            "tumor_fraction": r.get("tumor_fraction", 0),
        })

# ── Analysis B: SAM3 HELPED MedGemma ─────────────────────────────────────────
# standalone wrong AND pipeline right

helped = []
for r in sam3:
    gt  = gt_class(r)
    pp  = pipeline_pred(r)
    sp  = standalone_pred(r)
    if sp is None:
        continue
    standalone_wrong = (sp != gt)
    pipeline_right   = (pp == gt)
    if standalone_wrong and pipeline_right:
        helped.append({
            "image_path":      r.get("image_path"),
            "gt":              gt,
            "gt_sub":          gt_subclass(r),
            "standalone_pred": sp,
            "pipeline_pred":   pp,
            "tumor_fraction":  r.get("tumor_fraction", 0),
        })

# ── matched totals (images present in both results) ───────────────────────────

matched = [r for r in sam3 if standalone_pred(r) is not None]
n_matched = len(matched)

# both right, both wrong
both_right = [r for r in matched if pipeline_pred(r) == gt_class(r) and standalone_pred(r) == gt_class(r)]
both_wrong = [r for r in matched if pipeline_pred(r) != gt_class(r) and standalone_pred(r) != gt_class(r)]

# ── print report ──────────────────────────────────────────────────────────────

pr("=" * 65)
pr("  SAM3 INFLUENCE CROSS-REFERENCE ANALYSIS")
pr("=" * 65)
pr(f"\n  Matched images (present in both results): {n_matched}")
pr(f"  Both correct:                             {len(both_right)}  ({pct(len(both_right), n_matched):.1f}%)")
pr(f"  Both wrong:                               {len(both_wrong)}  ({pct(len(both_wrong), n_matched):.1f}%)")
pr(f"  SAM3 HURT  (standalone✓ → pipeline✗):    {len(hurt)}  ({pct(len(hurt), n_matched):.1f}%)")
pr(f"  SAM3 HELPED (standalone✗ → pipeline✓):   {len(helped)}  ({pct(len(helped), n_matched):.1f}%)")

# ── breakdown of HURT cases ───────────────────────────────────────────────────

pr("\n" + "=" * 65)
pr("  A) SAM3 HURT — standalone correct but pipeline wrong")
pr("     (SAM3 bounding box misled MedGemma)")
pr("=" * 65)
pr(f"  Total: {len(hurt)}")

# by gt class
hurt_tumor  = [h for h in hurt if h["gt"] == "tumor"]
hurt_normal = [h for h in hurt if h["gt"] == "normal"]
pr(f"\n  Ground-truth tumor:  {len(hurt_tumor)}")
pr(f"  Ground-truth normal: {len(hurt_normal)}")

# for normal images that got hurt: what did pipeline predict?
pr(f"\n  Normal images hurt — pipeline predicted:")
pp_counts = defaultdict(int)
for h in hurt_normal:
    pp_counts[h["pipeline_pred"]] += 1
for pred, cnt in sorted(pp_counts.items(), key=lambda x: -x[1]):
    pr(f"    → {pred:<25} {cnt}")

# for tumor images that got hurt: what did pipeline predict?
pr(f"\n  Tumor images hurt — pipeline predicted:")
pp_counts_t = defaultdict(int)
for h in hurt_tumor:
    pp_counts_t[h["pipeline_pred"]] += 1
for pred, cnt in sorted(pp_counts_t.items(), key=lambda x: -x[1]):
    pr(f"    → {pred:<25} {cnt}")

# by subclass (tumor subtypes that got hurt)
pr(f"\n  Hurt tumor images by subtype (gt_sub → pipeline_pred):")
sub_counts = defaultdict(lambda: defaultdict(int))
for h in hurt_tumor:
    sub = h["gt_sub"] or "tumor"
    sub_counts[sub][h["pipeline_pred"]] += 1
for sub in sorted(sub_counts):
    total = sum(sub_counts[sub].values())
    preds = ", ".join(f"{p}({n})" for p, n in sorted(sub_counts[sub].items(), key=lambda x: -x[1]))
    pr(f"    {sub:<20} n={total:<4} → {preds}")

pr("\n  Full list:")
for h in hurt:
    pr(f"  gt={h['gt']:<7} sub={str(h['gt_sub']):<15} "
       f"standalone={h['standalone_pred']:<10} pipeline={h['pipeline_pred']:<10} "
       f"frac={h['tumor_fraction']:.5f}  {h['image_path']}")

# ── breakdown of HELPED cases ─────────────────────────────────────────────────

pr("\n" + "=" * 65)
pr("  B) SAM3 HELPED — standalone wrong but pipeline correct")
pr("     (SAM3 bounding box corrected a MedGemma mistake)")
pr("=" * 65)
pr(f"  Total: {len(helped)}")

helped_tumor  = [h for h in helped if h["gt"] == "tumor"]
helped_normal = [h for h in helped if h["gt"] == "normal"]
pr(f"\n  Ground-truth tumor:  {len(helped_tumor)}")
pr(f"  Ground-truth normal: {len(helped_normal)}")

pr(f"\n  What standalone had predicted (before SAM3 correction):")
sp_counts = defaultdict(int)
for h in helped:
    sp_counts[h["standalone_pred"]] += 1
for pred, cnt in sorted(sp_counts.items(), key=lambda x: -x[1]):
    pr(f"    standalone predicted {pred:<25} → corrected to gt ({cnt} cases)")

pr(f"\n  Helped tumor images by subtype:")
sub_helped = defaultdict(lambda: defaultdict(int))
for h in helped_tumor:
    sub = h["gt_sub"] or "tumor"
    sub_helped[sub][h["standalone_pred"]] += 1
for sub in sorted(sub_helped):
    total = sum(sub_helped[sub].values())
    preds = ", ".join(f"was_{p}({n})" for p, n in sorted(sub_helped[sub].items(), key=lambda x: -x[1]))
    pr(f"    {sub:<20} n={total:<4} {preds}")

pr("\n  Full list:")
for h in helped:
    pr(f"  gt={h['gt']:<7} sub={str(h['gt_sub']):<15} "
       f"standalone={h['standalone_pred']:<10} pipeline={h['pipeline_pred']:<10} "
       f"frac={h['tumor_fraction']:.5f}  {h['image_path']}")

# ── save text report ──────────────────────────────────────────────────────────

with open(OUT_TXT, "w") as f:
    f.write("\n".join(_lines))
print(f"\n[saved] {OUT_TXT}")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 11: overall 2x2 outcome summary ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["Both\ncorrect", "SAM3\nhelped", "SAM3\nhurt", "Both\nwrong"]
values = [len(both_right), len(helped), len(hurt), len(both_wrong)]
colors = [GREEN, BLUE, RED, GRAY]
bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{v}\n({pct(v, n_matched):.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Image count")
ax.set_ylim(0, max(values) * 1.25)
ax.set_title(f"SAM3 influence on MedGemma diagnosis\n(n={n_matched} matched images)")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "11_sam3_hurt_vs_helped.png")

# ── Plot 12: HURT breakdown — by gt class and pipeline prediction ─────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# left: hurt normal images — what pipeline predicted
ax = axes[0]
if hurt_normal:
    pp_n = defaultdict(int)
    for h in hurt_normal:
        pp_n[h["pipeline_pred"]] += 1
    labels_n = list(pp_n.keys())
    vals_n   = [pp_n[l] for l in labels_n]
    ax.bar(labels_n, vals_n, color=RED, edgecolor="white", width=0.5)
    for i, v in enumerate(vals_n):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10)
ax.set_title(f"Hurt: normal images (n={len(hurt_normal)})\nStandalone✓ normal → Pipeline✗ predicted:")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)

# right: hurt tumor images — by subtype
ax = axes[1]
if hurt_tumor:
    sub_n = defaultdict(int)
    for h in hurt_tumor:
        sub_n[h["gt_sub"] or "tumor"] += 1
    labels_t = sorted(sub_n.keys(), key=lambda x: -sub_n[x])
    vals_t   = [sub_n[l] for l in labels_t]
    ax.barh(labels_t[::-1], vals_t[::-1], color=RED, edgecolor="white", height=0.6)
    for i, v in enumerate(vals_t[::-1]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
ax.set_title(f"Hurt: tumor images by subtype (n={len(hurt_tumor)})\nStandalone✓ tumor → Pipeline✗")
ax.set_xlabel("Count")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("SAM3 HURT cases — breakdown", fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "12_sam3_hurt_breakdown.png")

# ── Plot 13: HELPED breakdown — what standalone had predicted ─────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# left: helped tumor images — what standalone had wrongly predicted
ax = axes[0]
if helped_tumor:
    sp_t = defaultdict(int)
    for h in helped_tumor:
        sp_t[h["standalone_pred"]] += 1
    labels_ht = sorted(sp_t.keys(), key=lambda x: -sp_t[x])
    vals_ht   = [sp_t[l] for l in labels_ht]
    ax.bar(labels_ht, vals_ht, color=BLUE, edgecolor="white", width=0.5)
    for i, v in enumerate(vals_ht):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10)
ax.set_title(f"Helped: tumor images (n={len(helped_tumor)})\nStandalone✗ predicted → Pipeline✓ tumor")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=20)
ax.spines[["top", "right"]].set_visible(False)

# right: helped tumor images — by subtype
ax = axes[1]
if helped_tumor:
    sub_h = defaultdict(int)
    for h in helped_tumor:
        sub_h[h["gt_sub"] or "tumor"] += 1
    labels_sh = sorted(sub_h.keys(), key=lambda x: -sub_h[x])
    vals_sh   = [sub_h[l] for l in labels_sh]
    ax.barh(labels_sh[::-1], vals_sh[::-1], color=BLUE, edgecolor="white", height=0.6)
    for i, v in enumerate(vals_sh[::-1]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
ax.set_title(f"Helped: tumor images by subtype (n={len(helped_tumor)})\nStandalone✗ → Pipeline✓")
ax.set_xlabel("Count")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("SAM3 HELPED cases — breakdown", fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "13_sam3_helped_breakdown.png")

print(f"\nDone. Report: {OUT_TXT} | Plots: {OUT_PLOTS}/")
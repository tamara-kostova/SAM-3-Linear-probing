"""
Simple metrics for standalone MedGemma + SAM3 influence analysis.
Usage: python compute_metrics.py

Outputs:
  results/metrics_report.txt  – full text report
  results/plots/                     – PNG visualisations
"""
import json, os
from collections import defaultdict
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STANDALONE  = "results/phases/standalone_summary_results.jsonl"
SAM3        = "results/sam3_summary_results.jsonl"
OUT_TXT     = "results/metrics_report.txt"
OUT_PLOTS   = "results/plots"
os.makedirs(OUT_PLOTS, exist_ok=True)

FIELDS = [
    ("modality",             "input.metadata.modality",         "output.parsed_response.modality"),
    ("specialized_sequence", "input.metadata.modality_subtype", "output.parsed_response.specialized_sequence"),
    ("plane",                "input.metadata.axial_plane",      "output.parsed_response.plane"),
    ("diagnosis_name",       "input.metadata.class",            "output.parsed_response.diagnosis_name"),
    ("diagnosis_detailed",   "input.metadata.subclass",         "output.parsed_response.diagnosis_detailed"),
]
SCORE_KEYS   = ["diagnosis_confidence", "severity_score", "severity_confidence"]
TUMOR_THRESH = 0.001
TUMOR_DATASETS = {"images-17", "images-44c", "figshare", "br35h"}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get(rec, path):
    obj = rec
    for k in path.split("."):
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj

def pct(c, t):
    return c / t * 100 if t else 0.0

def avg_floats(vals):
    v = [float(x) for x in vals if x is not None and x != "null"]
    return (mean(v), len(v)) if v else (None, 0)

def acc_str(c, t):
    return f"{c}/{t} = {pct(c,t):.1f}%" if t else "N/A"

# ── tee: write to stdout AND collect for file ─────────────────────────────────

_lines = []

def pr(s=""):
    print(s)
    _lines.append(s)

# ── compute field accuracy ────────────────────────────────────────────────────

def field_accuracy(records, gt_path, pred_path):
    """Returns (overall_correct, overall_total, {value: [correct, total]})."""
    by_val = defaultdict(lambda: [0, 0])
    for r in records:
        gt = get(r, gt_path)
        if gt is None:
            continue
        key = str(gt).lower()
        by_val[key][1] += 1
        pred = get(r, pred_path)
        if pred is not None and str(pred).lower() == key:
            by_val[key][0] += 1
    total_c = sum(v[0] for v in by_val.values())
    total_t = sum(v[1] for v in by_val.values())
    return total_c, total_t, dict(by_val)

# ── PART 1: Standalone MedGemma ───────────────────────────────────────────────

records = [r for r in load_jsonl(STANDALONE)
           if (r.get("input", {}).get("metadata", {}).get("dataset") or "").lower()
              in TUMOR_DATASETS]

pr("\n" + "="*60)
pr("  STANDALONE MEDGEMMA — Accuracy per field")
pr("="*60)

field_results = {}   # field -> (total_c, total_t, by_val)
for field, gt_path, pred_path in FIELDS:
    tc, tt, by_val = field_accuracy(records, gt_path, pred_path)
    field_results[field] = (tc, tt, by_val)
    pr(f"\n[{field}]  overall: {acc_str(tc, tt)}")
    for val in sorted(by_val):
        c, t = by_val[val]
        pr(f"    {val:<30} {acc_str(c, t)}")

# ── Score distributions ───────────────────────────────────────────────────────

pr("\n" + "="*60)
pr("  STANDALONE MEDGEMMA — Score distributions")
pr("="*60)

score_data = {}   # sk -> {"overall": (avg, n), "by_dx": {dx: (avg, n)}}
for sk in SCORE_KEYS:
    overall_vals = [get(r, f"output.parsed_response.{sk}") for r in records]
    ov_avg, ov_n = avg_floats(overall_vals)
    by_dx = defaultdict(list)
    for r in records:
        dx  = get(r, "output.parsed_response.diagnosis_name")
        val = get(r, f"output.parsed_response.{sk}")
        if dx:
            by_dx[str(dx).lower()].append(val)
    by_dx_avg = {dx: avg_floats(v) for dx, v in by_dx.items()}
    score_data[sk] = {"overall": (ov_avg, ov_n), "by_dx": by_dx_avg}

    pr(f"\n[{sk}]  overall avg: {f'{ov_avg:.3f} (n={ov_n})' if ov_avg else 'N/A'}")
    for dx in sorted(by_dx_avg):
        a, n = by_dx_avg[dx]
        pr(f"    {dx:<30} {f'{a:.3f} (n={n})' if a else 'N/A'}")

# ── PART 2: SAM3 + MedGemma accuracy ─────────────────────────────────────────

sam3 = [r for r in load_jsonl(SAM3) if r.get("mode") == "sam3_bbox"]

# Build path lookup from standalone so we can fill in gt modality/plane/sequence
def norm_path(p):
    for prefix in ("data/raw data/", "data/"):
        if str(p or "").startswith(prefix):
            return str(p)[len(prefix):]
    return str(p or "")

_meta_lookup = {}
for r in records:
    ip = get(r, "input.image_path") or ""
    meta = get(r, "input.metadata") or {}
    _meta_lookup[norm_path(ip)] = meta

# Normalize each SAM3 record to the same nested structure field_accuracy expects
def norm_sam3(r):
    pred = r.get("prediction") or {}
    gt_label = r.get("gt_label") or "normal"
    gt_class  = "normal" if gt_label == "normal" else "tumor"
    gt_sub    = None if gt_label == "normal" else gt_label
    meta = _meta_lookup.get(norm_path(r.get("image_path", "")), {})
    return {
        "input": {"metadata": {
            "class":            gt_class,
            "subclass":         gt_sub,
            "modality":         meta.get("modality"),
            "axial_plane":      meta.get("axial_plane"),
            "modality_subtype": meta.get("modality_subtype"),
        }},
        "output": {"parsed_response": {
            "modality":             pred.get("modality"),
            "specialized_sequence": pred.get("specialized_sequence"),
            "plane":                pred.get("plane"),
            "diagnosis_name":       pred.get("diagnosis_name"),
            "diagnosis_detailed":   pred.get("diagnosis_detailed"),
            "diagnosis_confidence": pred.get("diagnosis_confidence"),
            "severity_score":       pred.get("severity_score"),
            "severity_confidence":  pred.get("severity_confidence"),
        }},
    }

sam3_norm = [norm_sam3(r) for r in sam3]

pr("\n" + "="*60)
pr("  SAM3 + MEDGEMMA — Accuracy per field")
pr("="*60)

sam3_field_results = {}
for field, gt_path, pred_path in FIELDS:
    tc, tt, by_val = field_accuracy(sam3_norm, gt_path, pred_path)
    sam3_field_results[field] = (tc, tt, by_val)
    pr(f"\n[{field}]  overall: {acc_str(tc, tt)}")
    for val in sorted(by_val):
        c, t = by_val[val]
        pr(f"    {val:<30} {acc_str(c, t)}")

# ── SAM3 Score distributions ──────────────────────────────────────────────────

pr("\n" + "="*60)
pr("  SAM3 + MEDGEMMA — Score distributions")
pr("="*60)

sam3_score_data = {}
for sk in SCORE_KEYS:
    overall_vals = [get(r, f"output.parsed_response.{sk}") for r in sam3_norm]
    ov_avg, ov_n = avg_floats(overall_vals)
    by_dx = defaultdict(list)
    for r in sam3_norm:
        dx  = get(r, "output.parsed_response.diagnosis_name")
        val = get(r, f"output.parsed_response.{sk}")
        if dx:
            by_dx[str(dx).lower()].append(val)
    by_dx_avg = {dx: avg_floats(v) for dx, v in by_dx.items()}
    sam3_score_data[sk] = {"overall": (ov_avg, ov_n), "by_dx": by_dx_avg}

    pr(f"\n[{sk}]  overall avg: {f'{ov_avg:.3f} (n={ov_n})' if ov_avg else 'N/A'}")
    for dx in sorted(by_dx_avg):
        a, n = by_dx_avg[dx]
        pr(f"    {dx:<30} {f'{a:.3f} (n={n})' if a else 'N/A'}")

# ── PART 3: SAM3 influence ────────────────────────────────────────────────────

def is_tumor(r):  return r.get("gt_label", "normal") != "normal"
def pred_cls(r):   return (r.get("prediction") or {}).get("diagnosis_name", "")

tumor_imgs  = [r for r in sam3 if is_tumor(r)]
normal_imgs = [r for r in sam3 if not is_tumor(r)]
no_tumor    = [r for r in tumor_imgs if r.get("tumor_fraction", 1) < TUMOR_THRESH]
missed      = [r for r in no_tumor   if pred_cls(r) != "tumor"]
fp_seg      = [r for r in normal_imgs if r.get("tumor_fraction", 0) >= TUMOR_THRESH]
false_pos   = [r for r in fp_seg     if pred_cls(r) == "tumor"]

pr("\n" + "="*60)
pr("  SAM3 — Tumor images where SAM3 missed tumor → wrong pred")
pr("="*60)
pr(f"  Count: {len(missed)}")
for r in missed:
    pr(f"  gt={r['gt_label']:<15} pred={pred_cls(r):<10} "
       f"fraction={r.get('tumor_fraction',0):.5f}  {r['image_path']}")

pr("\n" + "="*60)
pr("  SAM3 — Normal images where SAM3 found tumor → pred=tumor")
pr("="*60)
pr(f"  Count: {len(false_pos)}")
for r in false_pos:
    pr(f"  gt=normal          pred=tumor          "
       f"fraction={r.get('tumor_fraction',0):.5f}  {r['image_path']}")

pr("\n" + "="*60)
pr("  SAM3 — Summary")
pr("="*60)
pr(f"  sam3_bbox records total : {len(sam3)}")
pr(f"  Tumor images            : {len(tumor_imgs)}")
pr(f"    SAM3 missed tumor     : {len(no_tumor)}")
pr(f"    → also mis-classified : {len(missed)}")
pr(f"  Normal images           : {len(normal_imgs)}")
pr(f"    SAM3 found tumor pxls : {len(fp_seg)}")
pr(f"    → also pred as tumor  : {len(false_pos)}")

# ── Save text report ──────────────────────────────────────────────────────────

with open(OUT_TXT, "w") as f:
    f.write("\n".join(_lines))
print(f"\n[saved] {OUT_TXT}")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

BLUE   = "#4C72B0"
GREEN  = "#55A868"
RED    = "#C44E52"
ORANGE = "#DD8452"
GRAY   = "#8C8C8C"
COLORS = [BLUE, GREEN, ORANGE, RED, "#9172B0", "#64B5CD", "#CCB974"]

def save(fig, name):
    p = os.path.join(OUT_PLOTS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")

# ── Plot 1: Overall accuracy per field ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
labels = [f for f, *_ in FIELDS]
vals   = [pct(field_results[f][0], field_results[f][1]) for f in labels]
bars   = ax.barh(labels[::-1], vals[::-1], color=BLUE, edgecolor="white", height=0.6)
for bar, v in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%", va="center", fontsize=10)
ax.set_xlim(0, 110)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Standalone MedGemma — Overall accuracy per field")
ax.axvline(50, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "01_field_accuracy_overview.png")

# ── Plot 2: Per-class accuracy for diagnosis_name ────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
by_val = field_results["diagnosis_name"][2]
classes = sorted(by_val)
accs    = [pct(by_val[c][0], by_val[c][1]) for c in classes]
counts  = [by_val[c][1] for c in classes]
bar_colors = [GREEN if a >= 60 else RED for a in accs]
bars = ax.barh(classes[::-1], accs[::-1], color=bar_colors[::-1],
               edgecolor="white", height=0.6)
for bar, v, n in zip(bars, accs[::-1], counts[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%  (n={n})", va="center", fontsize=9)
ax.set_xlim(0, 120)
ax.set_xlabel("Accuracy (%)")
ax.set_title("diagnosis_name — per-class accuracy")
ax.axvline(50, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "02_diagnosis_name_per_class.png")

# ── Plot 3: Per-class accuracy for diagnosis_detailed ────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
by_val  = field_results["diagnosis_detailed"][2]
classes = sorted(by_val)
accs    = [pct(by_val[c][0], by_val[c][1]) for c in classes]
counts  = [by_val[c][1] for c in classes]
bar_colors = [GREEN if a >= 30 else RED for a in accs]
bars = ax.barh(classes[::-1], accs[::-1], color=bar_colors[::-1],
               edgecolor="white", height=0.6)
for bar, v, n in zip(bars, accs[::-1], counts[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%  (n={n})", va="center", fontsize=8.5)
ax.set_xlim(0, 110)
ax.set_xlabel("Accuracy (%)")
ax.set_title("diagnosis_detailed — per-class accuracy")
ax.axvline(30, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "03_diagnosis_detailed_per_class.png")

# ── Plot 4: Score averages by predicted diagnosis_name ───────────────────────

all_dx = sorted({dx for sk in SCORE_KEYS
                    for dx in score_data[sk]["by_dx"]})
x = np.arange(len(all_dx))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, (sk, color) in enumerate(zip(SCORE_KEYS, [BLUE, ORANGE, GREEN])):
    vals = [score_data[sk]["by_dx"].get(dx, (None, 0))[0] or 0 for dx in all_dx]
    ax.bar(x + i*width, vals, width, label=sk, color=color, edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(all_dx, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean score")
ax.set_ylim(0, 1.1)
ax.set_title("Score averages by predicted diagnosis_name")
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "04_score_distributions.png")

# ── Plot 5: specialized_sequence per-class accuracy ──────────────────────────

fig, ax = plt.subplots(figsize=(7, 3.5))
by_val  = field_results["specialized_sequence"][2]
classes = sorted(by_val)
accs    = [pct(by_val[c][0], by_val[c][1]) for c in classes]
counts  = [by_val[c][1] for c in classes]
bars = ax.bar(classes, accs, color=BLUE, edgecolor="white", width=0.5)
for bar, v, n in zip(bars, accs, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 45)
ax.set_ylabel("Accuracy (%)")
ax.set_title("specialized_sequence (modality_subtype) — per-class accuracy")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "05_specialized_sequence_accuracy.png")

# ── Plot 6: SAM3 influence summary ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

categories = ["Tumor images\n(SAM3 missed)", "Normal images\n(SAM3 false seg)"]
total_bars = [len(no_tumor),  len(fp_seg)]
wrong_bars = [len(missed),    len(false_pos)]

x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, total_bars, w, label="SAM3 segmentation error",  color=ORANGE, edgecolor="white")
b2 = ax.bar(x + w/2, wrong_bars, w, label="→ also MedGemma wrong",    color=RED,    edgecolor="white")

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(h),
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# reference totals as dashed lines
ax.axhline(len(tumor_imgs),  color=BLUE,  linestyle="--", linewidth=1,
           label=f"Total tumor images ({len(tumor_imgs)})",  alpha=0.7, xmin=0.02, xmax=0.48)
ax.axhline(len(normal_imgs), color=GREEN, linestyle="--", linewidth=1,
           label=f"Total normal images ({len(normal_imgs)})", alpha=0.7, xmin=0.52, xmax=0.98)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel("Image count")
ax.set_title("SAM3 segmentation errors and their influence on MedGemma")
ax.legend(fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "06_sam3_influence.png")

# ── Plot 7: tumor_fraction distribution for SAM3 false positives (normal→tumor) ──

fracs_fp = [r.get("tumor_fraction", 0) for r in false_pos]
fracs_ok = [r.get("tumor_fraction", 0) for r in normal_imgs if pred_cls(r) != "tumor"]

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, max(fracs_fp + fracs_ok + [0.001]), 30)
ax.hist(fracs_ok, bins=bins, color=GREEN, alpha=0.6, label="Normal → correctly normal")
ax.hist(fracs_fp, bins=bins, color=RED,   alpha=0.8, label="Normal → wrongly predicted tumor")
ax.set_xlabel("tumor_fraction (SAM3)")
ax.set_ylabel("Count")
ax.set_title("SAM3 tumor fraction: normal images — correct vs false-positive predictions")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "07_sam3_fp_fraction_dist.png")

# ── Plot 8: Standalone vs SAM3 — overall accuracy per field (side-by-side) ───

fig, ax = plt.subplots(figsize=(9, 5))
field_labels = [f for f, *_ in FIELDS]
sa_vals  = [pct(field_results[f][0],      field_results[f][1])      for f in field_labels]
s3_vals  = [pct(sam3_field_results[f][0], sam3_field_results[f][1]) for f in field_labels]

x = np.arange(len(field_labels))
w = 0.35
b1 = ax.bar(x - w/2, sa_vals, w, label="Standalone MedGemma", color=BLUE,   edgecolor="white")
b2 = ax.bar(x + w/2, s3_vals, w, label="SAM3 + MedGemma",     color=ORANGE, edgecolor="white")
for bar, v in [(b, v) for bars, vals in [(b1, sa_vals), (b2, s3_vals)] for b, v in zip(bars, vals)]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(field_labels, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_title("Standalone MedGemma vs SAM3 + MedGemma — accuracy per field")
ax.axhline(50, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "08_standalone_vs_sam3_comparison.png")

# ── Plot 9: diagnosis_name per-class — Standalone vs SAM3 ────────────────────

all_classes = sorted(set(field_results["diagnosis_name"][2]) |
                     set(sam3_field_results["diagnosis_name"][2]))
sa_by = field_results["diagnosis_name"][2]
s3_by = sam3_field_results["diagnosis_name"][2]
sa_acc = [pct(sa_by.get(c, [0,0])[0], sa_by.get(c, [0,1])[1]) for c in all_classes]
s3_acc = [pct(s3_by.get(c, [0,0])[0], s3_by.get(c, [0,1])[1]) for c in all_classes]

x = np.arange(len(all_classes))
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - w/2, sa_acc, w, label="Standalone", color=BLUE,   edgecolor="white")
b2 = ax.bar(x + w/2, s3_acc, w, label="SAM3",       color=ORANGE, edgecolor="white")
for bar, v in [(b, v) for bars, vals in [(b1, sa_acc), (b2, s3_acc)] for b, v in zip(bars, vals)]:
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(all_classes, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_title("diagnosis_name — Standalone vs SAM3 + MedGemma")
ax.axhline(50, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "09_diagnosis_name_standalone_vs_sam3.png")

# ── Plot 10: diagnosis_detailed per-class — Standalone vs SAM3 ───────────────

all_sub = sorted(set(field_results["diagnosis_detailed"][2]) |
                 set(sam3_field_results["diagnosis_detailed"][2]))
sa_by = field_results["diagnosis_detailed"][2]
s3_by = sam3_field_results["diagnosis_detailed"][2]
sa_acc = [pct(sa_by.get(c, [0,0])[0], sa_by.get(c, [0,1])[1]) for c in all_sub]
s3_acc = [pct(s3_by.get(c, [0,0])[0], s3_by.get(c, [0,1])[1]) for c in all_sub]

x = np.arange(len(all_sub))
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, sa_acc, w, label="Standalone", color=BLUE,   edgecolor="white")
b2 = ax.bar(x + w/2, s3_acc, w, label="SAM3",       color=ORANGE, edgecolor="white")
for bar, v in [(b, v) for bars, vals in [(b1, sa_acc), (b2, s3_acc)] for b, v in zip(bars, vals)]:
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(all_sub, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_title("diagnosis_detailed — Standalone vs SAM3 + MedGemma")
ax.axhline(30, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "10_diagnosis_detailed_standalone_vs_sam3.png")

print(f"\nDone. Plots saved to {OUT_PLOTS}/")

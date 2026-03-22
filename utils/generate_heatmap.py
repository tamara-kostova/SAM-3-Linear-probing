"""
generate_heatmap.py
-------------------
Generates a paper-quality heatmap of dataset class distribution.

Outputs:
  dataset_heatmap.pdf  – vector, use this for paper submission
  dataset_heatmap.png  – raster preview

Usage:
  python generate_heatmap.py
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = "results/standalone_counts_per_dataset_class.csv"
DATASET_ORDER = ["Figshare", "Images-44c", "Images-17", "Br35H"]
CLASS_ORDER = [
    "Glioma",
    "Meningioma",
    "Pituitary Tumor",
    "Carcinoma",
    "Germinoma",
    "Granuloma",
    "Meduloblastoma",
    "Neurocitoma",
    "Papiloma",
    "Schwannoma",
    "Tuberculoma",
    "Other Abnorm.",
    "Normal",
    "Tumor (unlabeled)",
]

DATASET_CANON = {
    "figshare": "Figshare",
    "images-44c": "Images-44c",
    "images-17": "Images-17",
    "br35h": "Br35H",
}

CLASS_CANON = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary tumor": "Pituitary Tumor",
    "carcinoma": "Carcinoma",
    "germinoma": "Germinoma",
    "granuloma": "Granuloma",
    "meduloblastoma": "Meduloblastoma",
    "neurocitoma": "Neurocitoma",
    "papiloma": "Papiloma",
    "schwannoma": "Schwannoma",
    "tuberculoma": "Tuberculoma",
    "other abnormalities": "Other Abnorm.",
    "normal": "Normal",
    "tumor": "Tumor (unlabeled)",
}

def title_case_label(s):
    if s is None:
        return ""
    s = str(s).strip()
    return s.title()

def canon_dataset(s):
    if s is None:
        return ""
    key = str(s).strip().lower()
    return DATASET_CANON.get(key, title_case_label(s))

def canon_class(s):
    if s is None:
        return ""
    key = str(s).strip().lower()
    return CLASS_CANON.get(key, title_case_label(s))

def load_counts(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")

    header = rows[0]
    if len(header) < 3 or header[0].lower() != "dataset":
        raise ValueError("CSV header must start with 'dataset'")

    class_labels = header[1:-1]
    data_rows = [r for r in rows[1:] if r and r[0].upper() != "TOTAL"]

    datasets = []
    data = []
    for r in data_rows:
        if not r:
            continue
        ds = r[0]
        vals = [int(v) if v else 0 for v in r[1:-1]]
        datasets.append(canon_dataset(ds))
        data.append(vals)

    data = np.array(data).T
    classes = [canon_class(c) for c in class_labels]

    # Reorder datasets to desired order; keep any extras at the end
    ds_to_idx = {d: i for i, d in enumerate(datasets)}
    ordered_datasets = [d for d in DATASET_ORDER if d in ds_to_idx]
    ordered_datasets += [d for d in datasets if d not in ordered_datasets]
    data = data[:, [ds_to_idx[d] for d in ordered_datasets]]

    # Reorder classes to desired order; include missing classes as zero rows
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    ordered_classes = [c for c in CLASS_ORDER]
    extra_classes = [c for c in classes if c not in ordered_classes]
    ordered_classes += extra_classes

    # Build reordered data with zero rows for missing classes
    reordered = []
    for c in ordered_classes:
        if c in cls_to_idx:
            reordered.append(data[cls_to_idx[c], :])
        else:
            reordered.append(np.zeros(data.shape[1], dtype=int))
    data = np.vstack(reordered)

    return ordered_datasets, ordered_classes, data

DATASETS, CLASSES, DATA = load_counts(INPUT_CSV)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 7))

# YlOrRd: perceptually uniform, works in grayscale print
cmap = plt.cm.YlOrRd
cmap.set_under('white')   # zero cells render white

im = ax.imshow(DATA, cmap=cmap, aspect='auto', vmin=1, vmax=1600)

# ── Axis labels ───────────────────────────────────────────────────────────────
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels(DATASETS, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(CLASSES)))
ax.set_yticklabels(CLASSES, fontsize=9)

# Put dataset names on top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# ── Cell annotations ──────────────────────────────────────────────────────────
for i in range(len(CLASSES)):
    for j in range(len(DATASETS)):
        v = DATA[i, j]
        if v > 0:
            color = 'white' if v > 900 else '#222222'
            ax.text(j, i, f'{v:,}',
                    ha='center', va='center',
                    fontsize=8, color=color)
        else:
            ax.text(j, i, '—',
                    ha='center', va='center',
                    fontsize=9, color='#bbbbbb')

# ── Grid lines between cells ──────────────────────────────────────────────────
for x in np.arange(-0.5, len(DATASETS), 1):
    ax.axvline(x, color='white', linewidth=1.5)
for y in np.arange(-0.5, len(CLASSES), 1):
    ax.axhline(y, color='white', linewidth=1.5)

# Dashed separator between tumor subtypes and normal/unlabeled rows
ax.axhline(11.5, color='#444444', linewidth=1.2, linestyle='--')

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
cbar.set_label('Image count', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── Title & layout ────────────────────────────────────────────────────────────
ax.set_title('Dataset composition by class',
             fontsize=11, fontweight='bold', pad=14)

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('results/plots/dataset_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

print("Saved: results/plots/dataset_heatmap.png")
print(f"Total images: {DATA.sum():,}")
print(f"Per dataset:  {dict(zip(DATASETS, DATA.sum(axis=0)))}")
print(f"Per class:    {dict(zip(CLASSES, DATA.sum(axis=1)))}")

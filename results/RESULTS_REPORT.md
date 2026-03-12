# SAM3 Linear Probing — Results Report

---

## 1. Overview

This project evaluates SAM3 (frozen backbone) across three neuroimaging tasks:
- **Tumor** — brain tumour segmentation (BraTS 2021)
- **MS** — multiple sclerosis lesion segmentation (MSLesSeg)
- **Stroke** — ischaemic infarct segmentation (ISLES 2022)

And a downstream task:
- **MedGemma diagnosis** — does SAM3 soft masking improve MedGemma's tumour classification?

All linear probes use a single `Conv2d(feature_dim → 2, kernel=1)` trained on top of the
**frozen** SAM3 image encoder. The backbone is never updated.

---

## 2. Task 1 — Brain Tumour Segmentation (BraTS)

### Zero-shot SAM3 (baseline)
| Metric | Value |
|--------|-------|
| Dice   | 0.450 |
| IoU    | 0.306 |

### Linear probe — 125 BraTS 2021 patients (non-overlapping test set)
| Metric    | Value     |
|-----------|-----------|
| Accuracy  | **0.9933** |
| IoU       | **0.7185** |
| Dice      | **0.8362** |
| Precision | 0.8811    |
| Recall    | 0.7957    |
| F1        | 0.8362    |

Confusion matrix (pixels): TN=9,919,504,318 · FP=23,390,097 · FN=44,493,271 · TP=173,252,314

**Improvement: +0.386 Dice over zero-shot.**

The probe was trained and validated on **BraTS 2020** (369 patients, `t1ce` modality,
80/20 train/val split) and tested on **125 non-overlapping BraTS 2021 patients**
(deduplicated against BraTS 2020 via TCIA metadata to prevent data leakage).
Results from `test_125_from_intersection/results.txt`.
Checkpoint: `checkpoints/final_probe.pth`.

### Why it worked well
- Large dataset (~1 251 patients, thousands of slices)
- T1ce-enhancing tumours are large and high-contrast
- SAM3 features cleanly separate tumour from background with a linear decision boundary

---

## 3. Task 2 — MS Lesion Segmentation (MSLesSeg)

### Zero-shot SAM3 — corrected protocol (test set, all slices, global pixel metrics)
Notebook: `12_SAM3_MS_dataset_all_slices.ipynb` — run 2026-03-10, prompt: "white matter lesion"

| Metric      | Value  |
|-------------|--------|
| Dice        | **0.052** |
| IoU         | 0.027  |
| Accuracy    | 0.999  |
| Sensitivity | 0.027  |
| Specificity | 1.000  |

Dataset: 22 test cases, 4 004 slices total (1 751 with lesion, 2 253 empty).

<details>
<summary>Original zero-shot (lesion slices only, best slice per case — not comparable)</summary>

| Metric | Value | Note |
|--------|-------|------|
| Dice   | 0.535 | Best lesion slice per case, per-slice Dice averaged |

</details>

### Linear probe — retrained with `empty_ratio=1.0` (test set, all slices)
| Metric   | Value      |
|----------|------------|
| Accuracy | 0.9834     |
| IoU      | **0.1511** |
| Dice     | **0.2625** |

**The retrained probe outperforms the corrected zero-shot SAM3 (+0.210 Dice).**

The original zero-shot Dice of 0.535 was evaluated only on the best lesion slice per
case; when re-run on all slices with global pixel-level metrics (matching the linear
probe protocol), zero-shot drops to 0.052. The retrained probe's 0.263 Dice is a
genuine and substantially larger improvement.

<details>
<summary>Earlier probe trained with <code>empty_ratio=0.3</code> (distribution mismatch — superseded)</summary>

| Metric   | Value |
|----------|-------|
| Accuracy | 0.985 |
| IoU      | 0.081 |
| Dice     | 0.149 |

Probe trained on 30% empty slices, tested on 30% — understates real-world performance
because the test distribution did not match the ~89% empty-slice rate of the full dataset.

</details>

### Why the probe still underperforms supervised baselines
1. **Small dataset** — MSLesSeg has only ~75 patients (~22 train, ~53 test in provided split)
2. **Severe class imbalance** — MS lesions cover ~1–2% of pixels per slice; a linear
   probe with ~512 parameters cannot learn a reliable threshold under this imbalance.
3. **Feature resolution** — SAM3 downsamples to a coarse feature map; tiny MS plaques
   (sometimes 2–5 mm) may fall within a single feature vector after downsampling.

---

## 4. Task 3 — Stroke Lesion Segmentation (ISLES 2022)

### Zero-shot SAM3 (original notebook — best slice per case, 25/100 cases)
| Metric | Value | Note |
|--------|-------|------|
| Dice        | 0.492 | Best slice per case only |
| IoU         | 0.366 | Best slice per case only |
| Sensitivity | 0.487 | |
| Specificity | 0.996 | |
| Best prompt | "stroke lesion" | |

ISLES 2022 challenge winner (nnU-Net): **0.58 Dice**

### Linear probe — retrained with `empty_ratio=1.0` (test set, all slices)
| Metric   | Value      |
|----------|------------|
| Accuracy | 0.9865     |
| IoU      | **0.2564** |
| Dice     | **0.4082** |

**With matched train/test distribution the probe is competitive with zero-shot SAM3.**

The zero-shot result (0.492 Dice) was evaluated only on the best lesion slice per case
(25/100 patients). A corrected all-slices evaluation
(`14_SAM3_Stroke_dataset_all_slices.ipynb`) was too slow to run — comparison remains
protocol-mismatched until that notebook is re-run.

<details>
<summary>Earlier probe trained with <code>empty_ratio=0.3</code> (validation set only — superseded)</summary>

| Metric    | Value |
|-----------|-------|
| Best IoU  | 0.125 |
| Best Dice | 0.224 |
| Final Acc | 0.952 |

Validation-only run, 30% empty slices — substantially underestimates performance.

</details>

### Why the probe may still underperform supervised baselines
- Stroke lesions have higher size variance (lacunar to large MCA territory)
- Only 100 ISLES patients — fewer than needed for reliable linear probe training
- DWI modality (used for stroke) may be less represented in SAM3's pretraining

---

## 5. Task 4 — MedGemma Tumour Diagnosis (SAM3 influence)

MedGemma (`medgemma-1.5-4b-it` via vLLM) was evaluated standalone and with SAM3
soft masking + bounding box on a multi-source brain MRI dataset
(figshare, images-17, images-44c, Br35H). Source: `results/metrics_report.txt`.

### Standalone MedGemma
| Field | Accuracy | n |
|-------|----------|---|
| Modality | **90.4%** | 14 957 |
| Diagnosis name | **80.8%** | 14 957 |
| Diagnosis detailed | 27.3% | 10 615 |
| Specialized sequence | 24.8% | 11 957 |
| Plane (axial) | 24.4% | 7 416 |
| Diagnosis confidence (avg score) | 0.948 | 14 749 |

Diagnosis name breakdown:
- Tumour: 85.2% (10 327 / 12 115)
- Normal: 67.1% (1 735 / 2 585)
- Other abnormalities: 6.6% (17 / 257)

Diagnosis detailed — strong on glioma only:
- Glioma: **69.8%** (2 752 / 3 940)
- Meningioma: 4.0% (115 / 2 888)
- Schwannoma: 2.7% (25 / 932)
- All other subtypes: 0.0%

### SAM3 + MedGemma (bounding box pipeline)
| Field | Standalone | SAM3 pipeline | Δ |
|-------|-----------|----------------|---|
| Modality | 90.4% | **98.1%** | +7.7 pp |
| Specialized sequence | 24.8% | **29.7%** | +4.9 pp |
| Plane (axial) | 24.4% | **51.0%** | **+26.6 pp** |
| Diagnosis name | 80.8% | **87.0%** | **+6.2 pp** |
| Diagnosis detailed | **27.3%** | 17.0% | −10.3 pp |
| Diagnosis confidence (avg) | **0.948** | 0.922 | −0.026 |

Diagnosis name with SAM3 (breakdown):
- Tumour: **96.3%** (9 612 / 9 978) — improved +11.1 pp
- Normal: 41.3% (845 / 2 046) — degraded −25.8 pp

Diagnosis detailed with SAM3:
- Glioma: **87.4%** (1 585 / 1 813) — improved +17.6 pp
- All other subtypes: ≤3.1% — collapsed (pipeline dominated by "tumour" prediction)

**Key insight:** The SAM3 bounding box concentrates MedGemma on the tumour region, strongly
boosting tumour and glioma detection. However its false positives cause large
drops for normal images and non-glioma pathologies.

### SAM3 influence cross-reference (11 038 matched images)
| Outcome | Count | % |
|---------|-------|---|
| Both correct | 8 288 | 75.1% |
| Both wrong | 779 | 7.1% |
| SAM3 **hurt** (standalone ✓ → pipeline ✗) | 727 | **6.6%** |
| SAM3 **helped** (standalone ✗ → pipeline ✓) | 1 244 | **11.3%** |

Net effect: SAM3 helped **+4.7 pp** more cases than it hurt.

Hurt cases breakdown:
- Normal images falsely predicted as tumour: **513 / 577** hurt normals (88.9%)
- Tumour images mis-predicted: 150 (mainly schwannoma, meningioma, non-glioma types)

SAM3 missed the tumour entirely in **128 images** (fraction ≈ 0), causing
MedGemma to predict normal or other abnormality on confirmed tumour cases.

---

## 6. Summary Table

| Task | Zero-shot Dice | Probe Dice | Δ | Comparable? |
|------|---------------|------------|---|-------------|
| Tumor (BraTS) | 0.450 | **0.836** (IoU 0.719, Acc 0.993) | +0.386 | Yes (all slices, 125-patient test set) |
| MS (MSLesSeg) | 0.052 | **0.263** (IoU 0.151, Acc 0.983) | **+0.210** | Yes (corrected — all slices, global pixel, test set) |
| Stroke (ISLES) | 0.492† | **0.408** (IoU 0.256, Acc 0.987) | −0.084† | No† — protocol mismatch |

† Stroke zero-shot evaluated on best slice per case (25/100 cases); corrected all-slices
  evaluation pending (`14_SAM3_Stroke_dataset_all_slices.ipynb`). The gap is likely
  smaller or reversed once zero-shot is re-run on all slices.

---

## 7. Future Work & Retraining Plan

### 7.1 ~~Fix train/test distribution mismatch (MS + Stroke)~~ — DONE

Both probes retrained with `empty_ratio=1.0`. Results in `results/ms/results.json` and
`results/stroke/results.json`. MS Dice improved from 0.149 → 0.263; Stroke from 0.224 → 0.408.

### 7.2 Re-run zero-shot notebooks with corrected protocol

MS zero-shot re-run complete (`12_SAM3_MS_dataset_all_slices.ipynb`, 2026-03-10):
Dice=0.052 on all 4 004 test slices — now fully comparable with the linear probe.

Stroke all-slices zero-shot (`14_SAM3_Stroke_dataset_all_slices.ipynb`) was too slow
to complete — still pending. Zero-shot comparison for stroke remains protocol-mismatched.

### 7.3 Address class imbalance more aggressively

Current class weights use inverse-frequency weighting. For MS/stroke the imbalance
is ~50:1 to ~100:1, which overwhelms a linear probe. Improvements:
- Increase Dice loss weight (`dice_weight` in `combined_loss`, currently 0.5 → try 1.0–2.0)
- Use focal loss instead of cross-entropy for the CE component
- Oversample lesion slices rather than subsampling empty ones

### 7.4 Deeper probe for MS and Stroke

A single `Conv2d(1×1)` may lack capacity for tiny lesion tasks. Try a shallow MLP probe:
- 2-layer: `Conv2d(C→64, k=1) → ReLU → Conv2d(64→2, k=1)`
- Still linear in the strict sense per spatial location, but with one hidden layer
- Expected improvement: captures non-linear feature combinations that a single linear
  threshold cannot

### 7.5 Larger MS dataset

MSLesSeg (~75 patients) is too small. Consider combining with:
- **MSSEG 2016** (53 patients, FLAIR + T1 + T2 + FLAIR-DP + FLAIR-PD)
- **WMH Challenge 2017** (60 patients, white matter hyperintensity)
- Both are publicly available; the MSLesSeg loader already supports FLAIR modality

### 7.6 Stroke: switch to DWI modality consistently

The zero-shot notebook used FLAIR for stroke (what was available in the ISLES BIDS
`anat/` folder), but the linear probe used DWI (`--stroke_modality dwi`). For a fair
comparison both should use the same modality. DWI is the clinical standard for acute
stroke and likely gives better feature separation.

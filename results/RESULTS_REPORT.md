# SAM3 Linear Probing -- Results Report

---

## 1. Overview

This report summarizes:
- SAM3 zero-shot segmentation on BraTS 2021
- Linear probing on frozen SAM3 features (BraTS 2020 train, BraTS 2021 test)
- Generalisation to MS and stroke segmentation
- MedGemma 1.5 4B standalone MRI classification
- SAM3 + MedGemma spatial-attention pipeline and its failure modes

All probes use a single `1x1 Conv2d` head on top of the frozen SAM3 image encoder.

---

## 2. SAM3 Zero-Shot Segmentation (BraTS 2021)

Evaluation on 125 BraTS 2021 patients (T1ce), pixel-level aggregate with 95% CI:

| Metric | Value | 95% CI |
|---|---:|---:|
| Dice | 0.189 | [0.184, 0.193] |
| IoU | 0.124 | [0.120, 0.128] |
| Sensitivity | 0.397 | [0.389, 0.405] |

SAM3 produces non-empty masks on all cases and shows partial tumor localization
(sensitivity 0.397), but precision is poor and Dice/IoU are too low for clinical use.

---

## 3. Linear Probing on Frozen SAM3 Features (BraTS)

**Setup**: Linear probe trained on BraTS 2020, evaluated on the same 125 BraTS 2021
patients as the zero-shot test set.

**Results** (mean with 95% CI):

| Method | Dice | IoU | Sensitivity | Precision |
|---|---:|---:|---:|---:|
| Zero-shot (pixel) | 0.189 | 0.124 | 0.397 | --- |
| Probe (pixel) | **0.836** | **0.719** | **0.796** | **0.881** |
| Probe (case mean) | **0.801** | **0.693** | --- | --- |
| Delta (pixel) | **+0.647** | **+0.595** | **+0.399** | --- |

Per-case Dice mean is 0.801 with 95% CI [0.770, 0.832]. The per-case IoU mean is
0.693 with 95% CI [0.662, 0.725].

**Statistical test (paired, n=125):**
- Shapiro-Wilk rejected normality of differences (p < 1e-4).
- One-sided Wilcoxon signed-rank test: W = 7863.0, p = 1.98e-22.

Conclusion: the linear probe improvement over zero-shot is highly significant.

---

## 4. Generalisation: MS and Stroke Segmentation

**MSLesSeg (MS lesions)**
- Zero-shot SAM3 (prompt: "white matter lesion"): Dice = 0.052, IoU = 0.027, Acc = 0.999
- Linear probe: Dice = 0.263, IoU = 0.151, Acc = 0.983

**ISLES 2022 (stroke lesions, DWI)**
- Zero-shot SAM3 (prompt: "ischemic infarct"):
  - Per-case Dice = 0.107 (95% CI [0.061, 0.154])
  - Pixel-level Dice = 0.342, IoU = 0.207
- Linear probe: Dice = 0.408, IoU = 0.256, Acc = 0.987

Performance is substantially lower than BraTS due to small datasets, severe class
imbalance, and SAM3 feature downsampling that can blur tiny lesions.

---

## 5. MedGemma 1.5 4B Standalone (Zero-Shot)

All tasks evaluated with a structured neuroradiology prompt (temperature 0).
Accuracies with Wilson 95% CIs are reported in the paper; table below is the
point-estimate summary.

| Task | Subclass | Correct / Total | Acc. |
|---|---|---:|---:|
| Modality | MRI (all) | 12,114 / 13,390 | 90.5% |
| Sequence | T1 | 340 / 2,439 | 13.9% |
| Sequence | T1c+ | 1,548 / 5,706 | 27.1% |
| Sequence | T2 | 662 / 2,248 | 29.4% |
| Sequence | Overall | 2,550 / 10,393 | 24.5% |
| Plane | Axial | 1,653 / 5,852 | 28.2% |
| Diagnosis | Normal | 1,735 / 2,584 | 67.1% |
| Diagnosis | Tumor | 9,192 / 10,806 | 85.1% |
| Diagnosis | Overall | 10,927 / 13,390 | 81.6% |
| Subtype | Glioma | 2,752 / 3,940 | 69.8% |
| Subtype | Meningioma | 84 / 1,586 | 5.3% |
| Subtype | Schwannoma | 25 / 928 | 2.7% |
| Subtype | Pituitary | 0 / 930 | 0.0% |
| Subtype | Overall | 2,862 / 9,309 | 30.7% |

MedGemma is strong on coarse tasks (modality, tumor vs normal) but weak on
fine-grained attributes (sequence, plane, subtype). A strong glioma bias dominates
subtype prediction. Mean diagnosis confidence is 0.948, but calibration is not measured.

---

## 6. SAM3 + MedGemma Pipeline

SAM3 proposes a bounding box that is overlaid on the full image before MedGemma
inference. Base pipeline uses theta = 0.0 (no confidence threshold).

**Threshold comparison (diagnosis task):**

| Configuration | Tumor Acc (%) | Normal Acc (%) |
|---|---:|---:|
| Standalone MedGemma | 85.1 | 67.1 |
| Pipeline (theta = 0.00) | 96.3 | 41.3 |

**Pipeline vs standalone (matched images; McNemar p-values):**

| Task | Standalone (%) | Pipeline (%) | Delta | p |
|---|---:|---:|---:|---:|
| Modality | 90.5 | 98.1 | +7.6 | <0.001 |
| Sequence | 24.5 | 29.5 | +5.0 | <0.001 |
| Image Plane | 28.2 | 51.6 | +23.4 | <0.001 |
| Diagnosis (overall) | 81.6 | 87.0 | +5.4 | <0.001 |
| Diagnosis (tumor) | 85.1 | 96.3 | +11.2 | <0.001 |
| Diagnosis (normal) | 67.1 | 41.3 | -25.8 | <0.001 |
| Subtype (overall) | 30.7 | 17.0 | -13.7 | <0.001 |

Key failure modes:
- Normal-scan specificity collapse due to SAM3 false positives.
- Subtype accuracy degradation; bounding boxes bias MedGemma toward generic "tumor".
- Missed-tumor cases when SAM3 outputs no mask.

---

## 7. Case-Level SAM3 Influence (Diagnosis Task)

Matched images: n = 10,119.

| Outcome | Count | % |
|---|---:|---:|
| Both correct | 7,510 | 74.2% |
| Both wrong | 767 | 7.6% |
| SAM3 helped (standalone wrong -> pipeline correct) | 1,133 | 11.2% |
| SAM3 hurt (standalone correct -> pipeline wrong) | 709 | 7.0% |

Net benefit: +424 images in favor of the pipeline.

Hurt cases breakdown:
- 577 hurt normal images; 513 (88.9%) flipped to tumor.
- 132 hurt tumor images; most redirected to other abnormalities or normal.

---

## 8. Summary Table

| Task | Zero-shot Dice | Probe Dice | Delta | Comparable? |
|---|---:|---:|---:|---|
| Tumor (BraTS) | 0.189 | **0.836** | +0.647 | Yes (same 125-patient cohort) |
| MS (MSLesSeg) | 0.052 | **0.263** | +0.210 | Yes (all slices, global pixel) |
| Stroke (ISLES) | 0.107 (per-case) | **0.408** | +0.301 | Mixed (per-case vs pixel) |

Note: Stroke zero-shot is reported as per-case Dice, while the probe reports pixel-level
Dice. Both are included in the paper with explicit protocol notes.

---

## 9. Future Work (as stated in the paper)

- Sweep SAM3 confidence threshold theta in {0.10, 0.30, 0.50} to reduce false positives.
- Insert a normal/abnormal filter between SAM3 and MedGemma.
- Prompt engineering for rare subtype discrimination.
- Extend to multi-class BraTS sub-region segmentation.
- Formal calibration evaluation for MedGemma confidence (ECE).

---

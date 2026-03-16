#!/usr/bin/env python3
"""
SAM3 Zero-Shot Stroke Segmentation on ISLES 2022 — DWI modality, all slices.

Converted from 14_SAM3_Stroke_dataset_all_slices.ipynb.
Key changes vs original notebook:
  - FLAIR → DWI (to match the linear probe modality)
  - Best-slice-only → all slices (fixes protocol mismatch)
  - Per-case Dice saved to CSV for Wilcoxon comparison with probe

Usage:
    python zeroshot_stroke_sam3.py --data_root data/ISLES2022

    # Specify only your 25 test subjects:
    python zeroshot_stroke_sam3.py \
        --data_root     data/ISLES2022 \
        --subjects_file stroke_test_subjects.txt \
        --output_dir    results/zeroshot_stroke_dwi
"""

import os
import gc
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
import scipy.stats as st
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom
from sam3.sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PROMPT = "ischemic infarct"


# ── metrics — identical to notebook ──────────────────────────────────────────

def compute_metrics(pred, gt):
    pred_flat = pred.flatten() > 0.5
    gt_flat   = gt.flatten()   > 0.5
    tp = np.sum( pred_flat &  gt_flat)
    fp = np.sum( pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat &  gt_flat)
    tn = np.sum(~pred_flat & ~gt_flat)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou  = tp / (tp + fp + fn)         if (tp + fp + fn)     > 0 else 0.0
    sens = tp / (tp + fn)              if (tp + fn)           > 0 else 0.0
    spec = tn / (tn + fp)              if (tn + fp)           > 0 else 0.0
    return {'dice': dice, 'iou': iou, 'sensitivity': sens, 'specificity': spec,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def normalize_mri(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


# ── SAM3 inference─────────────────────────────────

def safe_sam3_inference(processor, img, prompt, debug=False):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        inference_state = processor.set_image(img)
        output          = processor.set_text_prompt(prompt, inference_state)

        masks  = output.get("masks",  torch.tensor([]))
        scores = output.get("scores", torch.tensor([]))

        if debug:
            print(f"  '{prompt}': {len(masks)} masks, "
                  f"scores={scores.tolist() if len(scores) > 0 else []}")

        if len(masks) == 0 or len(scores) == 0:
            return None, 0.0, 0

        best_idx   = torch.argmax(scores)
        best_mask  = masks[best_idx].cpu().numpy().squeeze()
        best_score = scores[best_idx].item()

        torch.cuda.empty_cache()
        return best_mask, best_score, len(masks)

    except Exception as e:
        if debug:
            print(f"  ⚠  Error: {e}")
        torch.cuda.empty_cache()
        return None, 0.0, 0


# ── data loading — DWI instead of FLAIR ──────────────────────────────────────

def load_stroke_case_dwi(case_name, base_path):
    """
    Load DWI volume + binary lesion mask for one ISLES subject.
    DWI is in ses-0001/dwi/; mask in derivatives/ — same as linear probe.
    """
    base_path = Path(base_path)
    dwi = mask = None

    # DWI image
    dwi_dir   = base_path / case_name / "ses-0001" / "dwi"
    dwi_files = sorted([
        f for f in dwi_dir.rglob("*.nii*")
        if f.is_file()
    ])

    if not dwi_files:
        print(f"  [WARN] No DWI in {dwi_dir}")
        return None, None
    
    dwi_vol = nib.load(dwi_files[0]).get_fdata()
    if dwi_vol.ndim == 4:                        # 4-D: take b=1000 (index 1)
        dwi_vol = dwi_vol[..., min(1, dwi_vol.shape[-1] - 1)]
    dwi = dwi_vol

    mask_dir   = base_path / "derivatives" / case_name / "ses-0001"
    mask_files = list(mask_dir.glob("*_msk.nii*"))
    if not mask_files:
        print(f"  [WARN] No mask in {mask_dir}")
        return None, None

    mask = nib.load(mask_files[0]).get_fdata()

    # Resample mask to DWI space if needed
    if dwi.shape != mask.shape:
        zoom_factors = [dwi.shape[i] / mask.shape[i] for i in range(3)]
        mask = zoom(mask, zoom_factors, order=0)
        if mask.shape != dwi.shape:
            result = np.zeros(dwi.shape)
            sl = tuple(slice(0, min(mask.shape[i], dwi.shape[i])) for i in range(3))
            result[sl] = mask[sl]
            mask = result

    return dwi.astype(np.float32), (mask > 0).astype(np.uint8)


def slice_to_pil(img_slice):
    norm = normalize_mri(img_slice)
    return Image.fromarray((norm * 255).astype(np.uint8)).convert('RGB')


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(subjects, data_root, processor, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    per_case_rows = []
    g_tp = g_fp = g_fn = g_tn = 0

    for case_name in tqdm(subjects, desc="Subjects"):
        dwi, mask_vol = load_stroke_case_dwi(case_name, data_root)
        if dwi is None:
            tqdm.write(f"  [SKIP] {case_name}")
            continue

        n_slices      = dwi.shape[2]
        slice_dices   = []
        slice_ious    = []
        no_mask_count = 0

        for s in range(n_slices):
            pil_img = slice_to_pil(dwi[:, :, s])
            gt      = mask_vol[:, :, s]

            pred, score, _ = safe_sam3_inference(processor, pil_img, PROMPT)

            if pred is None:
                pred = np.zeros_like(gt)
                no_mask_count += 1
            elif pred.shape != gt.shape:
                pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
                pred_pil = pred_pil.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                pred     = (np.array(pred_pil) > 127).astype(np.uint8)

            m = compute_metrics(pred, gt)
            slice_dices.append(m['dice'])
            slice_ious.append(m['iou'])
            g_tp += m['tp']; g_fp += m['fp']
            g_fn += m['fn']; g_tn += m['tn']

        case_dice = float(np.mean(slice_dices))
        case_iou  = float(np.mean(slice_ious))

        per_case_rows.append({
            'case':           case_name,
            'n_slices':       n_slices,
            'no_mask_slices': no_mask_count,
            'mean_dice':      round(case_dice, 6),
            'mean_iou':       round(case_iou,  6),
            'min_dice':       round(float(np.min(slice_dices)), 6),
            'max_dice':       round(float(np.max(slice_dices)), 6),
        })
        tqdm.write(f"  {case_name}: {n_slices} slices | "
                   f"mean_dice={case_dice:.3f} | no_mask={no_mask_count}")

        gc.collect()
        torch.cuda.empty_cache()

    # summary
    eps         = 1e-8
    global_dice = 2 * g_tp / (2 * g_tp + g_fp + g_fn + eps)
    global_iou  = g_tp / (g_tp + g_fp + g_fn + eps)
    dices       = np.array([r['mean_dice'] for r in per_case_rows])
    n           = len(dices)
    mean_d      = dices.mean()
    ci_lo, ci_hi = st.t.interval(0.95, df=n - 1, loc=mean_d, scale=st.sem(dices))

    print("\n" + "=" * 60)
    print("ZERO-SHOT STROKE RESULTS  (DWI, all slices)")
    print("=" * 60)
    print(f"Cases evaluated     : {n}")
    print(f"Per-case Dice       : {mean_d:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Std Dice            : {dices.std():.4f}")
    print(f"Median Dice         : {np.median(dices):.4f}")
    print(f"Global (pixel) Dice : {global_dice:.4f}")
    print(f"Global (pixel) IoU  : {global_iou:.4f}")
    print("=" * 60)

    # save CSV
    csv_path = os.path.join(output_dir, "per_case_dice.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case', 'n_slices', 'no_mask_slices',
            'mean_dice', 'mean_iou', 'min_dice', 'max_dice'
        ])
        writer.writeheader()
        writer.writerows(per_case_rows)
    print(f"✓ Per-case CSV      : {csv_path}")

    # save summary
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SAM3 Zero-Shot Stroke — DWI, all slices\n")
        f.write("=" * 60 + "\n")
        f.write(f"Prompt              : {PROMPT}\n")
        f.write(f"Cases evaluated     : {n}\n")
        f.write(f"Per-case Dice       : {mean_d:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"Std Dice            : {dices.std():.4f}\n")
        f.write(f"Median Dice         : {np.median(dices):.4f}\n")
        f.write(f"Global (pixel) Dice : {global_dice:.4f}\n")
        f.write(f"Global (pixel) IoU  : {global_iou:.4f}\n")
    print(f"✓ Summary           : {txt_path}")


def load_sam3(device="cuda"):

    print("Loading SAM3...")

    bpe_path = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    # build model
    model = build_sam3_image_model(
        bpe_path=bpe_path
    )

    model.to(device)
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=0.5)

    print("✓ SAM3 loaded")

    return model, processor
# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--output_dir",  default="results/zeroshot_stroke_dwi")
    p.add_argument("--device",      default="cuda")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--test_subjects", nargs="+")
    grp.add_argument("--subjects_file")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SAM3 (ultralytics SAM wrapper)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, processor = load_sam3(device)

    # Subject list
    if args.test_subjects:
        subjects = args.test_subjects
    elif args.subjects_file:
        with open(args.subjects_file) as f:
            subjects = [l.strip() for l in f if l.strip()]
    else:
        # auto-discover
        rawdata = Path(args.data_root) / "rawdata"
        if rawdata.is_dir():
            subjects = sorted(d for d in os.listdir(rawdata) if d.startswith("sub-"))
        else:
            subjects = sorted(
                d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith("sub-")
            )
        print(f"Auto-discovered {len(subjects)} subjects")

    print(f"Subjects to evaluate: {len(subjects)}")
    evaluate(subjects, args.data_root, processor, args.output_dir)


if __name__ == "__main__":
    main()
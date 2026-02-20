#!/usr/bin/env python3
"""
BraTS2021 Ground Truth Bounding Box — Oracle Experiment

This script establishes the UPPER BOUND of what segmentation can contribute
to MedGemma classification. It uses the ground truth segmentation mask
(_seg.nii.gz) to draw a perfect bounding box, so any gap between this result
and the SAM3 result is purely due to segmentation quality.

BraTS2021 label convention:
    0 = background
    1 = necrotic tumor core (NCR)
    2 = peritumoral edematous/invaded tissue (ED)
    4 = enhancing tumor (ET)

For each patient volume it:
    1. Loads the segmentation and one or more MRI modalities
    2. Finds the axial slice with the largest tumor area
    3. Normalises the slice to uint8
    4. Draws the ground truth bounding box (whole tumour = labels 1+2+4)
    5. Saves a PNG + manifest entry compatible with run_medgemma_eval.py

Usage:
    python run_brats_gt_bbox.py \
        --data_dir ./BraTS2021_Training_Data \
        --output_dir ./segmented_gt_bbox \
        --modality flair \
        --max_patients 200

Then feed straight into Phase 2:
    python run_medgemma_eval.py \
        --manifest ./segmented_gt_bbox/gt_bbox_manifest_<timestamp>.json \
        --output_dir ./results/oracle
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")


# ============================================================================
# BRATS LABEL CONSTANTS
# ============================================================================

LABEL_NCR  = 1   # Necrotic core
LABEL_ED   = 2   # Edema
LABEL_ET   = 4   # Enhancing tumour

# Whole tumour = all non-background labels
WHOLE_TUMOUR_LABELS = {LABEL_NCR, LABEL_ED, LABEL_ET}

# Sub-region colours for optional multi-label overlay
LABEL_COLORS = {
    LABEL_NCR: (255, 0,   0),    # red
    LABEL_ED:  (0,   255, 0),    # green
    LABEL_ET:  (0,   0,   255),  # blue
}

MODALITIES = ["flair", "t1", "t1ce", "t2"]


# ============================================================================
# NIBABEL HELPERS
# ============================================================================

def load_volume(patient_dir: Path, suffix: str) -> np.ndarray:
    """Load a .nii.gz volume and return a float32 numpy array [H, W, D]."""
    pattern = f"*_{suffix}.nii.gz"
    matches = list(patient_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {patient_dir}")
    img = nib.load(str(matches[0]))
    return img.get_fdata(dtype=np.float32)


def normalise_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Percentile-clip and scale a 2-D float slice to uint8."""
    lo = np.percentile(slice_2d, 1)
    hi = np.percentile(slice_2d, 99)
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    clipped = np.clip(slice_2d, lo, hi)
    scaled  = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def best_axial_slice(seg_vol: np.ndarray) -> int:
    """Return the axial index (along last axis) with the most tumour voxels."""
    tumour_mask = np.isin(seg_vol, list(WHOLE_TUMOUR_LABELS))
    counts = tumour_mask.sum(axis=(0, 1))   # sum over H and W for each slice
    if counts.max() == 0:
        return seg_vol.shape[2] // 2        # fallback: mid-slice
    return int(counts.argmax())


def bounding_box_from_mask(binary_mask: np.ndarray):
    """
    Return (x1, y1, x2, y2) pixel bounding box of a 2-D binary mask.
    Returns None if mask is empty.
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][[0, -1]])
    x1, x2 = int(np.where(cols)[0][[0, -1]])
    return x1, y1, x2, y2


# ============================================================================
# IMAGE BUILDING
# ============================================================================

def build_gt_bbox_image(
    mri_slice:   np.ndarray,   # [H, W] float
    seg_slice:   np.ndarray,   # [H, W] int, BraTS labels
    mode:        str = "whole",
    box_color:   tuple = (255, 255, 0),
    thickness:   int = 3,
    show_labels: bool = False,
) -> Image.Image:
    """
    Build a PIL RGB image with the ground-truth bounding box overlaid.

    mode:
        "whole"   — single box around the whole tumour (labels 1+2+4)
        "subregion" — one box per sub-region in its own colour
    show_labels:
        If True, also draw a semi-transparent filled region for each label
        (useful for visual sanity checking).
    """
    # Convert MRI slice to RGB
    grey = normalise_slice(mri_slice)
    rgb  = np.stack([grey, grey, grey], axis=-1)
    img  = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if mode == "whole":
        binary = np.isin(seg_slice, list(WHOLE_TUMOUR_LABELS))

        if show_labels:
            # Tinted overlay per sub-region
            for label, color in LABEL_COLORS.items():
                region = (seg_slice == label)
                if not region.any():
                    continue
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                ys, xs = np.where(region)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    od.point((x, y), fill=(*color, 80))
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(img)

        bbox = bounding_box_from_mask(binary)
        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=thickness)

    elif mode == "subregion":
        for label, color in LABEL_COLORS.items():
            region = (seg_slice == label)
            if not region.any():
                continue
            bbox = bounding_box_from_mask(region)
            if bbox:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

    return img


# ============================================================================
# PATIENT LABEL DERIVATION
# ============================================================================

def derive_gt_label(seg_vol: np.ndarray) -> str:
    """
    Derive a string label from the segmentation volume.
    BraTS training data only contains gliomas (HGG/LGG), but we record
    tumour sub-type presence so downstream metrics are meaningful.
    """
    has_et  = (seg_vol == LABEL_ET).any()
    has_ncr = (seg_vol == LABEL_NCR).any()
    has_ed  = (seg_vol == LABEL_ED).any()
    has_any = has_et or has_ncr or has_ed

    if not has_any:
        return "normal"
    # ET present → likely HGG / glioblastoma
    if has_et:
        return "glioma_hgg"
    # NCR/ED only → likely LGG
    return "glioma_lgg"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_gt_bbox(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = args.modality if isinstance(args.modality, list) else [args.modality]

    # Collect patient directories
    patient_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS2021_")
    ])

    if not patient_dirs:
        raise RuntimeError(f"No BraTS2021_* directories found under {data_dir}")

    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]

    print(f"\n{'='*80}")
    print(f"BraTS2021 GT BOUNDING BOX  —  {len(patient_dirs)} patients")
    print(f"Modalities : {modalities}")
    print(f"Box mode   : {args.box_mode}")
    print(f"Output     : {output_dir}")
    print("="*80 + "\n")

    manifest = []
    skipped  = 0

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        patient_id = patient_dir.name

        try:
            # Load segmentation
            seg_vol = load_volume(patient_dir, "seg")
            seg_vol = seg_vol.astype(np.int32)

            # Best slice
            slice_idx = best_axial_slice(seg_vol)
            seg_slice = seg_vol[:, :, slice_idx]       # [H, W]

            # Ground-truth label
            gt_label = derive_gt_label(seg_vol)

            # Tumour stats for the chosen slice
            tumour_mask_slice = np.isin(seg_slice, list(WHOLE_TUMOUR_LABELS))
            tumour_pixels     = int(tumour_mask_slice.sum())
            total_pixels      = int(seg_slice.size)

            bbox = bounding_box_from_mask(tumour_mask_slice)

            # One output image per requested modality
            for mod in modalities:
                try:
                    mri_vol   = load_volume(patient_dir, mod)
                    mri_slice = mri_vol[:, :, slice_idx]    # [H, W]
                except FileNotFoundError as e:
                    print(f"\n  Skipping modality {mod} for {patient_id}: {e}")
                    continue

                img = build_gt_bbox_image(
                    mri_slice  = mri_slice,
                    seg_slice  = seg_slice,
                    mode       = args.box_mode,
                    box_color  = tuple(args.box_color),
                    thickness  = args.thickness,
                    show_labels= args.show_labels,
                )

                # Resize for MedGemma (keeps aspect ratio)
                if max(img.size) > 512:
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)

                # Save
                save_dir = output_dir / gt_label / mod
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{patient_id}_{mod}_slice{slice_idx:03d}_gt_bbox.png"
                img.save(save_path)

                manifest.append({
                    "original_path"  : str(patient_dir / f"{patient_id}_{mod}.nii.gz"),
                    "masked_path"    : str(save_path),
                    "mask_path"      : None,             # GT, no .npy needed
                    "gt_label"       : gt_label,
                    "dataset"        : "brats2021",
                    "patient_id"     : patient_id,
                    "modality"       : mod,
                    "slice_index"    : int(slice_idx),
                    "bbox"           : list(bbox) if bbox else None,
                    "tumor_pixels"   : tumour_pixels,
                    "total_pixels"   : total_pixels,
                    "tumor_fraction" : float(tumour_pixels / total_pixels) if total_pixels else 0.0,
                    "mask_type"      : f"gt_bbox_{args.box_mode}",
                })

        except Exception as e:
            print(f"\nWarning: skipping {patient_id}: {e}")
            skipped += 1
            continue

    # Write manifest
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / f"gt_bbox_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[OK] Saved  : {len(manifest)} images  |  Skipped: {skipped} patients")
    print(f"[OK] Manifest: {manifest_path}")
    print("="*80)
    print("\nNext step — start vLLM then run:")
    print(f"  python run_medgemma_eval.py --manifest {manifest_path} --output_dir ./results/oracle")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BraTS2021 GT bounding box oracle for MedGemma evaluation"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing BraTS2021_* patient folders")
    parser.add_argument("--output_dir", default="./segmented_gt_bbox",
                        help="Where to save output images and manifest")
    parser.add_argument("--modality", nargs="+", default=["flair"],
                        choices=MODALITIES,
                        help="MRI modality/modalities to visualise (default: flair)")
    parser.add_argument("--box_mode", choices=["whole", "subregion"], default="whole",
                        help=(
                            "'whole' draws one box around all tumour labels; "
                            "'subregion' draws a separate coloured box per label"
                        ))
    parser.add_argument("--box_color", nargs=3, type=int, default=[255, 255, 0],
                        metavar=("R", "G", "B"),
                        help="RGB colour for the whole-tumour bounding box (default: 255 255 0)")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Bounding box line thickness in pixels")
    parser.add_argument("--show_labels", action="store_true",
                        help="Overlay semi-transparent sub-region colour fills")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Cap number of patients processed (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gt_bbox(args)
#!/usr/bin/env python3
"""
Phase 1: SAM3 Segmentation

Loads images from tumor datasets, runs SAM3 segmentation,
saves masked images to disk. Run this BEFORE starting vLLM.

Usage:
    python run_sam3_segmentation.py \
        --data_dirs ./tumor_datasets \
        --sam3_checkpoint ./checkpoints/final_probe.pth \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --output_dir ./segmented \
        --mask_type soft \
        --background_dim 0.3 \
        --max_samples 100
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import cv2


# ============================================================================
# UTILITIES
# ============================================================================

def _resolve_device(requested="auto"):
    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return req


def _normalize_probe_state_dict(ckpt):
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format for linear probe")

    if "weight" in state and "bias" in state:
        return {"weight": state["weight"], "bias": state["bias"]}
    if "classifier.weight" in state and "classifier.bias" in state:
        return {"weight": state["classifier.weight"], "bias": state["classifier.bias"]}
    if "module.classifier.weight" in state and "module.classifier.bias" in state:
        return {"weight": state["module.classifier.weight"], "bias": state["module.classifier.bias"]}

    raise ValueError(f"Unsupported probe checkpoint keys: {list(state.keys())[:5]}")


def _norm_label(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "pituitary": "pituitary_tumor",
        "pituitary_tumour": "pituitary_tumor",
        "papiloma": "papilloma",
        "astrocytoma": "glioma",
        "glioblastoma": "glioma",
        "oligodendroglioma": "glioma",
        "ependymoma": "glioma",
        "ganglioglioma": "glioma",
        "no_tumor": "normal",
        "notumor": "normal",
    }
    return aliases.get(text, text)


def load_samples_from_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list")

    samples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        if not image_path:
            continue
        dataset = item.get("dataset")
        gt_label = item.get("gt_label")
        if gt_label is None:
            gt_name = _norm_label(item.get("class"))
            gt_subclass = _norm_label(item.get("subclass"))
            gt_label = gt_subclass if gt_subclass else gt_name
        samples.append({
            "image_path": image_path,
            "dataset": dataset or "manifest",
            "gt_label": gt_label or "tumor",
        })
    return samples


# ============================================================================
# SAM3 SEGMENTOR
# ============================================================================

class SAM3Segmentor:
    def __init__(self, checkpoint_path, bpe_path, device="auto"):
        from sam3.sam3 import build_sam3_image_model

        print("Loading SAM3...")
        self.device = _resolve_device(device)

        sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=self.device)

        ckpt = torch.load(checkpoint_path, map_location=torch.device(self.device))
        feature_dim = ckpt.get("feature_dim", 256) if isinstance(ckpt, dict) else 256

        self.probe = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.probe.load_state_dict(_normalize_probe_state_dict(ckpt))

        self.sam3 = sam3_model.to(self.device)
        self.probe.to(self.device)
        self.probe.eval()

        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()

        print("[OK] SAM3 loaded")

    @torch.no_grad()
    def segment(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.shape[-1] != 3:
            image = image[..., :3]

        image = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(image).permute(2, 0, 1)

        orig_h, orig_w = image.shape[:2]
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), size=(1008, 1008),
            mode="bilinear", align_corners=False
        )
        img_tensor = img_tensor.to(self.device)

        features = self.sam3.backbone(img_tensor, ["brain tumor"])

        if isinstance(features, dict):
            for key in ["vision_features", "image_features", "features"]:
                if key in features:
                    features = features[key]
                    break

        logits = self.probe(features)
        logits = torch.nn.functional.interpolate(
            logits, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        )

        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        del img_tensor, features, logits
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return mask.astype(np.uint8)


# ============================================================================
# MASKING HELPERS
# ============================================================================

def apply_soft_mask(image, mask, background_dim=0.3):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = image.astype(np.float32) / 255.0
    mask_3c = np.expand_dims(mask, axis=-1)
    blended = image * mask_3c + image * (1 - mask_3c) * background_dim
    return (blended * 255).astype(np.uint8)


def apply_bounding_box(image, mask, color=(255, 0, 0), thickness=3):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    return image


def apply_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended.astype(np.uint8)


# ============================================================================
# DATASET LOADER
# ============================================================================

class TumorDatasetLoader:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.samples = []
        self._load_datasets()

    def _load_datasets(self):
        print(f"\n{'='*80}\nLOADING DATASETS\n{'='*80}")
        for data_dir in self.data_dirs:
            dataset_name = Path(data_dir).name
            print(f"\nScanning: {dataset_name}")
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(Path(data_dir).rglob(ext))
            print(f"  Found {len(image_files)} images")
            for img_path in image_files:
                self.samples.append({
                    "image_path": str(img_path),
                    "dataset": dataset_name,
                    "gt_label": self._extract_label(img_path),
                })
        print(f"\n[OK] Total samples: {len(self.samples)}\n{'='*80}")

    def _extract_label(self, img_path):
        path_str = str(img_path).lower()
        if any(x in path_str for x in ["glioma", "astrocytoma", "glioblastoma"]):
            return "glioma"
        elif "meningioma" in path_str:
            return "meningioma"
        elif "pituitary" in path_str:
            return "pituitary_tumor"
        elif any(x in path_str for x in ["normal", "no_tumor", "notumor"]):
            return "normal"
        parent = Path(img_path).parent.name.lower()
        if "glioma" in parent:
            return "glioma"
        elif "meningioma" in parent:
            return "meningioma"
        elif "pituitary" in parent:
            return "pituitary_tumor"
        elif "normal" in parent or "no" in parent:
            return "normal"
        return "tumor"


# ============================================================================
# SEGMENTATION PIPELINE
# ============================================================================

def run_segmentation(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_manifest:
        samples = load_samples_from_manifest(args.input_manifest)
        print(f"[OK] Loaded {len(samples)} samples from manifest")
    else:
        dataset = TumorDatasetLoader(args.data_dirs)
        samples = dataset.samples

    segmentor = SAM3Segmentor(args.sam3_checkpoint, args.bpe_path, args.device)
    if args.max_samples:
        samples = samples[:args.max_samples]

    manifest = []
    skipped = 0

    print(f"\n{'='*80}")
    print(f"SEGMENTING {len(samples)} IMAGES  →  {output_dir}")
    print(f"Mask type: {args.mask_type}")
    print("="*80 + "\n")

    for sample in tqdm(samples, desc="Segmenting"):
        img_path = sample["image_path"]
        gt_label = sample["gt_label"]

        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)

            mask = segmentor.segment(image)

            # Apply visual treatment
            if args.mask_type == "soft":
                result = apply_soft_mask(image, mask, args.background_dim)
                result_img = Image.fromarray(result)
            elif args.mask_type == "bbox":
                result_img = apply_bounding_box(image, mask)
            elif args.mask_type == "overlay":
                result = apply_overlay(image, mask)
                result_img = Image.fromarray(result)
            else:
                result_img = image

            # Save masked image — mirror original directory structure
            rel_path = Path(img_path).stem
            dataset_tag = sample["dataset"]
            save_dir = output_dir / dataset_tag / gt_label
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{rel_path}_masked.png"
            result_img.save(save_path)

            # Also save raw mask as npy for optional inspection
            mask_save_path = save_dir / f"{rel_path}_mask.npy"
            np.save(mask_save_path, mask)

            manifest.append({
                "original_path": img_path,
                "masked_path": str(save_path),
                "mask_path": str(mask_save_path),
                "gt_label": gt_label,
                "dataset": dataset_tag,
                "tumor_pixels": int(mask.sum()),
                "total_pixels": int(mask.size),
                "tumor_fraction": float(mask.sum() / mask.size),
                "mask_type": args.mask_type,
            })

        except Exception as e:
            print(f"\nWarning: skipping {img_path}: {e}")
            skipped += 1
            continue

    # Write manifest so Phase 2 knows exactly what to load
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / f"segmentation_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[OK] Segmented: {len(manifest)}  |  Skipped: {skipped}")
    print(f"[OK] Manifest:  {manifest_path}")
    print("="*80)
    print("\nYou can now start vLLM and run run_medgemma_eval.py")
    print(f"  --manifest {manifest_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: SAM3 Segmentation")
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Directories containing tumor datasets")
    parser.add_argument("--input_manifest", default=None,
                        help="JSON file with image records (from concat script)")
    parser.add_argument("--sam3_checkpoint", required=True,
                        help="Path to SAM3 linear probe checkpoint")
    parser.add_argument("--bpe_path", required=True,
                        help="Path to BPE vocabulary file")
    parser.add_argument("--output_dir", default="./segmented",
                        help="Where to save masked images and manifest")
    parser.add_argument("--mask_type", choices=["soft", "bbox", "overlay"], default="soft")
    parser.add_argument("--background_dim", type=float, default=0.3,
                        help="Background dimming factor for soft mask (0-1)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of images (for testing)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.data_dirs and not args.input_manifest:
        parser.error("Provide either --data_dirs or --input_manifest")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_segmentation(args)

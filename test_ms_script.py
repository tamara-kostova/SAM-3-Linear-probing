#!/usr/bin/env python3
"""
Test a trained MS linear probe on MSLesSeg test data.

Example:
  python test_ms_script.py \
    --test_root "MS/MSLesSeg Dataset/test" \
    --checkpoint ./results/probe_ms.pth \
    --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
    --save_dir ./results/ms_test
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset_loaders import MSLesSegDataset
from multi_probes import LinearProbe, evaluate_probe
from linear_probing_universal import (
    HDF5FeaturesDataset, SAM3FeatureExtractor, precompute_features_hdf5
)


def load_or_compute_test_features(dataset, h5_path, bpe_path, device, batch_size):
    if os.path.exists(h5_path):
        print(f"✓ Using existing test features: {h5_path}")
        return HDF5FeaturesDataset(h5_path)

    from sam3.sam3 import build_sam3_image_model

    print("\nComputing MS test features ...")
    sam3_model = build_sam3_image_model(bpe_path=bpe_path)
    extractor = SAM3FeatureExtractor(sam3_model, device=device)

    original_extract = extractor.extract_features

    def extract_with_caption(images, captions=None):
        if captions is None:
            captions = ["white matter lesion"] * images.shape[0]
        return original_extract(images, captions)

    extractor.extract_features = extract_with_caption

    precompute_features_hdf5(
        dataset=dataset,
        feature_extractor=extractor,
        save_path=h5_path,
        batch_size=batch_size,
        device=device,
    )

    del sam3_model, extractor
    torch.cuda.empty_cache()
    return HDF5FeaturesDataset(h5_path)


def load_probe(checkpoint_path, feature_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    ckpt_feature_dim = int(ckpt.get("feature_dim", feature_dim))

    probe = LinearProbe(feature_dim=ckpt_feature_dim, num_classes=2).to(device)
    probe.load_state_dict(state)
    probe.eval()
    return probe, ckpt_feature_dim


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("MS TEST EVALUATION")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Test root: {args.test_root}")
    print(f"Checkpoint: {args.checkpoint}")

    cache_dir = os.path.join(args.save_dir, "cache_ms_test")
    test_ds = MSLesSegDataset(
        data_root=args.test_root,
        cache_dir=cache_dir,
        modality=args.modality,
        img_size=args.img_size,
        empty_ratio=1.0,  # keep all slices for unbiased test metrics
        force_rebuild=args.rebuild_cache,
    )
    print(f"✓ Test slices: {len(test_ds)}")

    h5_path = os.path.join(args.save_dir, "ms_test_features.h5")
    feat_ds = load_or_compute_test_features(
        dataset=test_ds,
        h5_path=h5_path,
        bpe_path=args.bpe_path,
        device=device,
        batch_size=args.feat_batch_size,
    )

    probe, feature_dim = load_probe(
        checkpoint_path=args.checkpoint,
        feature_dim=int(feat_ds.feature_dim),
        device=device,
    )
    print(f"✓ Probe loaded (feature_dim={feature_dim})")

    loader = DataLoader(
        feat_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    acc, iou, dice = evaluate_probe(loader, probe, device=device)

    print(f"\n{'=' * 70}")
    print("MS TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Accuracy: {acc:.4f}")
    print(f"IoU:      {iou:.4f}")
    print(f"Dice:     {dice:.4f}")

    out_txt = os.path.join(args.save_dir, "ms_test_results.txt")
    out_json = os.path.join(args.save_dir, "ms_test_results.json")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MS test results\n")
        f.write(f"test_root={args.test_root}\n")
        f.write(f"checkpoint={args.checkpoint}\n")
        f.write(f"accuracy={acc:.6f}\n")
        f.write(f"iou={iou:.6f}\n")
        f.write(f"dice={dice:.6f}\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_root": args.test_root,
                "checkpoint": args.checkpoint,
                "accuracy": float(acc),
                "iou": float(iou),
                "dice": float(dice),
            },
            f,
            indent=2,
        )

    print(f"✓ Saved {out_txt}")
    print(f"✓ Saved {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_root", type=str, default="MS/MSLesSeg Dataset/test")
    p.add_argument("--checkpoint", type=str, default="./results/probe_ms.pth")
    p.add_argument("--bpe_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./results/ms_test")
    p.add_argument("--modality", type=str, default="flair", choices=["flair", "t1"])
    p.add_argument("--img_size", type=int, default=1008)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--feat_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)


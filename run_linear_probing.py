#!/usr/bin/env python3
"""
run_linear_probing.py
=====================
Orchestrates SAM3 linear probing across Tumor (BraTS), MS (MSLesSeg),
and Stroke (ISLES 2022).

Usage
-----
  # Run all three tasks separately (Architecture A – recommended):
  python run_linear_probing.py \
      --tasks tumor ms stroke \
      --tumor_root  /data/BraTS2021 \
      --ms_root     /data/MSLesSeg \
      --stroke_root /data/ISLES2022 \
      --bpe_path    sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
      --save_dir    ./results \
      --arch        separate

  # Run multi-head joint training (Architecture B):
  python run_linear_probing.py ... --arch multihead

  # Run only MS (if you already have tumor done):
  python run_linear_probing.py --tasks ms --ms_root /data/MSLesSeg ...
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ── project imports ──────────────────────────────────────────────
from dataset_loaders import MSLesSegDataset, ISLESStrokeDataset
from multi_probes import (
    LinearProbe, MultiHeadProbe,
    train_separate_probe, train_multihead_probe,
    evaluate_probe, plot_separate_results, print_comparison_table,
    TASK_NAMES,
)

# BraTS dataset from your existing code
# (import whichever version you use)
try:
    from linear_probing import PreprocessedBraTSDataset
    HAS_BRATS = True
except ImportError:
    HAS_BRATS = False


# ─────────────────────────────────────────────────────────────────
# FEATURE PRE-COMPUTATION  (shared with existing pipeline)
# ─────────────────────────────────────────────────────────────────

def get_or_compute_features(
    dataset, split_name, task_name, save_dir,
    bpe_path, device, batch_size=16, force=False
):
    """
    Load pre-computed HDF5 features or compute them with SAM3.
    Returns an HDF5FeaturesDataset.
    """
    from linear_probing_universal import (
        HDF5FeaturesDataset, SAM3FeatureExtractor,
        precompute_features_hdf5
    )
    h5_path = os.path.join(save_dir, f'{task_name}_{split_name}_features.h5')

    if not os.path.exists(h5_path) or force:
        print(f"\nComputing {task_name}/{split_name} features …")
        from sam3.sam3 import build_sam3_image_model
        sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        extractor  = SAM3FeatureExtractor(sam3_model, device=device)

        # Use task-specific caption hint for SAM3's language conditioning
        caption_map = {
            'tumor':  'brain tumor',
            'ms':     'white matter lesion',
            'stroke': 'ischemic infarct',
        }
        caption = caption_map.get(task_name, 'lesion')

        # Patch caption into extractor
        original_extract = extractor.extract_features
        def extract_with_caption(images, captions=None):
            if captions is None:
                captions = [caption] * images.shape[0]
            return original_extract(images, captions)
        extractor.extract_features = extract_with_caption

        precompute_features_hdf5(
            dataset, extractor, h5_path,
            batch_size=batch_size, device=device
        )

        del sam3_model, extractor
        torch.cuda.empty_cache()

    return HDF5FeaturesDataset(h5_path)


# ─────────────────────────────────────────────────────────────────
# DATASET FACTORY
# ─────────────────────────────────────────────────────────────────

def build_datasets(args):
    """Return dict of {task: (train_dataset, val_dataset, test_dataset_or_None)}."""
    import torch.utils.data as tud
    splits = {}
    split_gen = torch.Generator().manual_seed(42)

    def split_train_val_test(full_ds):
        n_total = len(full_ds)
        n_test = int(n_total * args.test_ratio)
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Invalid split: total={n_total}, val={n_val}, test={n_test}. "
                "Adjust --val_ratio/--test_ratio."
            )
        if n_val <= 0:
            raise ValueError(
                "Validation split is empty. Increase --val_ratio "
                f"(current {args.val_ratio})."
            )

        if n_test > 0:
            train_ds, val_ds, test_ds = tud.random_split(
                full_ds, [n_train, n_val, n_test], generator=split_gen
            )
        else:
            train_ds, val_ds = tud.random_split(
                full_ds, [n_train, n_val], generator=split_gen
            )
            test_ds = None
        return train_ds, val_ds, test_ds

    if 'tumor' in args.tasks:
        assert HAS_BRATS, "linear_probing.py not found in PYTHONPATH"
        full = PreprocessedBraTSDataset(
            data_root=args.tumor_root,
            cache_dir=os.path.join(args.save_dir, 'cache_tumor'),
            modality=args.tumor_modality,
            force_rebuild=args.rebuild_cache,
        )
        train_ds, val_ds, test_ds = split_train_val_test(full)
        splits['tumor'] = (train_ds, val_ds, test_ds)
        msg = f"✓ Tumor  — train={len(train_ds)}  val={len(val_ds)}"
        if test_ds is not None:
            msg += f"  test={len(test_ds)}"
        print(msg)

    if 'ms' in args.tasks:
        full = MSLesSegDataset(
            data_root=args.ms_root,
            cache_dir=os.path.join(args.save_dir, 'cache_ms'),
            modality=args.ms_modality,
            empty_ratio=args.ms_empty_ratio,
            force_rebuild=args.rebuild_cache,
        )
        train_ds, val_ds, test_ds = split_train_val_test(full)
        splits['ms'] = (train_ds, val_ds, test_ds)
        msg = f"✓ MS     — train={len(train_ds)}  val={len(val_ds)}"
        if test_ds is not None:
            msg += f"  test={len(test_ds)}"
        print(msg)

    if 'stroke' in args.tasks:
        full = ISLESStrokeDataset(
            data_root=args.stroke_root,
            cache_dir=os.path.join(args.save_dir, 'cache_stroke'),
            modality=args.stroke_modality,
            empty_ratio=args.stroke_empty_ratio,
            force_rebuild=args.rebuild_cache,
        )
        train_ds, val_ds, test_ds = split_train_val_test(full)
        splits['stroke'] = (train_ds, val_ds, test_ds)
        msg = f"✓ Stroke — train={len(train_ds)}  val={len(val_ds)}"
        if test_ds is not None:
            msg += f"  test={len(test_ds)}"
        print(msg)

    return splits


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAM3 LINEAR PROBING  |  arch={args.arch.upper()}")
    print(f"Tasks: {args.tasks}")
    print(f"{'='*70}\n")

    # ── Build raw image datasets ─────────────────────────────────
    print("── Loading datasets ─────────────────────────────────────")
    dataset_splits = build_datasets(args)

    # ── Pre-compute / load HDF5 features ────────────────────────
    print("\n── Feature extraction ───────────────────────────────────")
    feature_splits = {}   # task → (train_feat_ds, val_feat_ds, test_feat_ds_or_None)
    feature_dim = None

    for task, (train_ds, val_ds, test_ds) in dataset_splits.items():
        train_feat = get_or_compute_features(
            train_ds, 'train', task, args.save_dir,
            args.bpe_path, device, args.feat_batch_size
        )
        val_feat = get_or_compute_features(
            val_ds, 'val', task, args.save_dir,
            args.bpe_path, device, args.feat_batch_size
        )
        test_feat = None
        if test_ds is not None:
            test_feat = get_or_compute_features(
                test_ds, 'test', task, args.save_dir,
                args.bpe_path, device, args.feat_batch_size
            )

        feature_splits[task] = (train_feat, val_feat, test_feat)

        # All tasks share the same SAM3 feature dim
        if feature_dim is None:
            feature_dim = int(train_feat.feature_dim)

    print(f"\n✓ Feature dimension: {feature_dim}")

    # ── DataLoaders ──────────────────────────────────────────────
    loaders = {}
    test_loaders = {}
    for task, (train_feat, val_feat, test_feat) in feature_splits.items():
        loaders[task] = (
            DataLoader(train_feat, batch_size=args.batch_size,
                       shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(val_feat,   batch_size=args.batch_size,
                       shuffle=False, num_workers=4, pin_memory=True),
        )
        if test_feat is not None:
            test_loaders[task] = DataLoader(
                test_feat, batch_size=args.batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
        else:
            test_loaders[task] = None

    # ── Class weights (for imbalanced tasks) ─────────────────────
    # Build once from full raw dataset (cheap; runs on cached .pt files)
    class_weights = {}
    for task in args.tasks:
        if task == 'ms':
            cw = dataset_splits['ms'][0].dataset.compute_class_weights()
            class_weights['ms'] = cw.tolist()
            print(f"  MS class weights: {class_weights['ms']}")
        elif task == 'stroke':
            cw = dataset_splits['stroke'][0].dataset.compute_class_weights()
            class_weights['stroke'] = cw.tolist()
            print(f"  Stroke class weights: {class_weights['stroke']}")
        else:
            class_weights[task] = None   # BraTS: equal weights

    # ── Training ─────────────────────────────────────────────────
    histories = {}

    if args.arch == 'separate':
        print(f"\n── Training SEPARATE probes ─────────────────────────────")
        probes = {}
        for task in args.tasks:
            train_loader, val_loader = loaders[task]
            probe, history = train_separate_probe(
                train_loader, val_loader,
                feature_dim=feature_dim,
                task_name=task,
                num_epochs=args.epochs,
                lr=args.lr,
                class_weights=class_weights.get(task),
                use_dice=(task in ('ms', 'stroke')),
                device=device,
            )
            probes[task] = probe
            histories[task] = history

            if test_loaders.get(task) is not None:
                test_acc, test_iou, test_dice = evaluate_probe(
                    test_loaders[task], probe, device
                )
                history['test_accuracy'] = test_acc
                history['test_iou'] = test_iou
                history['test_dice'] = test_dice
                print(
                    f"  {task} TEST: acc={test_acc:.4f}  "
                    f"IoU={test_iou:.4f}  Dice={test_dice:.4f}"
                )

            # Save individual probe
            save_path = os.path.join(args.save_dir, f'probe_{task}.pth')
            torch.save({
                'model_state_dict': probe.state_dict(),
                'feature_dim': feature_dim,
                'task': task,
                'history': history,
            }, save_path)
            print(f"  ✓ Saved {save_path}")

    elif args.arch == 'multihead':
        print(f"\n── Training MULTI-HEAD probe ────────────────────────────")
        probe, histories = train_multihead_probe(
            loaders_dict=loaders,
            feature_dim=feature_dim,
            num_epochs=args.epochs,
            lr=args.lr,
            class_weights_dict=class_weights,
            use_dice=True,
            device=device,
        )
        save_path = os.path.join(args.save_dir, 'probe_multihead.pth')
        torch.save({
            'model_state_dict': probe.state_dict(),
            'feature_dim': feature_dim,
            'tasks': args.tasks,
            'histories': histories,
        }, save_path)
        print(f"  ✓ Saved {save_path}")

        # Optional held-out test evaluation (per task)
        for task in args.tasks:
            if test_loaders.get(task) is None:
                continue

            def _fwd(feats):
                return probe(feats, task)

            test_acc, test_iou, test_dice = evaluate_probe(
                test_loaders[task], probe, device, forward_fn=_fwd
            )
            histories[task]['test_accuracy'] = test_acc
            histories[task]['test_iou'] = test_iou
            histories[task]['test_dice'] = test_dice
            print(
                f"  {task} TEST: acc={test_acc:.4f}  "
                f"IoU={test_iou:.4f}  Dice={test_dice:.4f}"
            )

    # ── Results ──────────────────────────────────────────────────
    print_comparison_table(histories)
    plot_separate_results(
        histories,
        save_path=os.path.join(args.save_dir, 'all_results.png')
    )

    import json
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tasks_str = '_'.join(args.tasks)
    results_path = os.path.join(args.save_dir, f'results_{tasks_str}_{ts}.json')
    empty_ratio_map = {'ms': args.ms_empty_ratio, 'stroke': args.stroke_empty_ratio}
    for task, history in histories.items():
        if task in empty_ratio_map:
            history['empty_ratio'] = empty_ratio_map[task]
    with open(results_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"✓ Results saved to {results_path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='SAM3 Linear Probing – multi-task extension'
    )

    # Architecture
    p.add_argument('--arch', default='separate',
                   choices=['separate', 'multihead'],
                   help='separate = one probe per task; multihead = joint training')
    p.add_argument('--tasks', nargs='+', default=['tumor', 'ms', 'stroke'],
                   choices=['tumor', 'ms', 'stroke'])

    # Data roots
    p.add_argument('--tumor_root',  type=str, default=None)
    p.add_argument('--ms_root',     type=str, default=None)
    p.add_argument('--stroke_root', type=str, default=None)

    # Modalities
    p.add_argument('--tumor_modality',  default='t1ce',
                   choices=['flair', 't1', 't1ce', 't2'])
    p.add_argument('--ms_modality',     default='flair',
                   choices=['flair', 't1'])
    p.add_argument('--stroke_modality', default='dwi',
                   choices=['dwi', 'adc', 'flair'])

    # Dataset tuning
    p.add_argument('--ms_empty_ratio',     type=float, default=1.0,
                   help='Fraction of empty (no-lesion) slices to keep for MS')
    p.add_argument('--stroke_empty_ratio', type=float, default=1.0,
                   help='Fraction of empty slices to keep for Stroke')
    p.add_argument('--val_ratio', type=float, default=0.15,
                   help='Validation split ratio (0-1)')
    p.add_argument('--test_ratio', type=float, default=0.15,
                   help='Held-out test split ratio (0-1); 0 disables test split')

    # Paths
    p.add_argument('--bpe_path',  type=str, required=True)
    p.add_argument('--save_dir',  type=str, default='./results')

    # Training
    p.add_argument('--epochs',          type=int,   default=20)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--feat_batch_size', type=int,   default=4,
                   help='Batch size for SAM3 feature extraction (smaller = less VRAM)')
    p.add_argument('--rebuild_cache',   action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

#!/usr/bin/env python3
"""
SAM3 Linear Probing — Multi-Condition Extension
================================================
Covers two architectures for probing SAM3 across Tumor, MS, and Stroke:

  Architecture A — Separate Probes (recommended for linear probing)
      One independent Conv2d(feature_dim → 2) per pathology.
      Train each with its own script run (or sequentially here).
      Best for: clean ablation, paper-ready per-task numbers.

  Architecture B — Multi-Head Probe (joint training)
      One shared identity "trunk" (SAM3 is frozen), three task heads.
      All heads trained simultaneously on a mixed dataset.
      Best for: checking whether joint training helps, lower GPU cost.

Because SAM3 is FROZEN, the backbone is identical in both architectures.
The difference is purely whether the probe heads are trained independently
or jointly. The diagram below summarises:

    ┌────────────────────────────────────────────────────────┐
    │               Frozen SAM3 Backbone                     │
    │   Images → image encoder → features [B, C, H', W']    │
    └──────────────┬─────────────────────┬───────────────────┘
                   │                     │
         ┌─────────▼──────────┐   ┌──────▼──────────────────┐
         │  Architecture A    │   │  Architecture B          │
         │  (separate probes) │   │  (multi-head probe)      │
         │                    │   │                          │
         │  probe_tumor       │   │  head_tumor   ──► logits │
         │  probe_ms     (×3) │   │  head_ms      ──► logits │
         │  probe_stroke      │   │  head_stroke  ──► logits │
         └────────────────────┘   └──────────────────────────┘

RECOMMENDATION: Use Architecture A unless you have a specific reason to
jointly train (e.g. severe data imbalance across tasks). See comments.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import h5py
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────
# ARCHITECTURE A: SEPARATE PROBES
# ─────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    A single linear probe head.
    A Conv2d(1×1) is mathematically equivalent to a linear classifier
    applied independently to each spatial location.

    num_classes=2 for all three tasks (binary: lesion vs background).
    """
    def __init__(self, feature_dim: int, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, features):                # [B, C, H, W]
        return self.classifier(features)        # [B, num_classes, H, W]


# ─────────────────────────────────────────────────────────────────
# ARCHITECTURE B: MULTI-HEAD PROBE
# ─────────────────────────────────────────────────────────────────

TASK_NAMES = ('tumor', 'ms', 'stroke')

class MultiHeadProbe(nn.Module):
    """
    Three independent linear heads sharing the same (frozen) SAM3 features.

    Why this is still "linear probing":
      - The SAM3 backbone stays completely frozen.
      - Each head is a single Conv2d(1×1), i.e. a linear classifier.
      - The only thing that changes vs Architecture A is that all three
        heads are trained in the same forward pass on mixed batches.

    Joint training can help when:
      ✓ One dataset is much smaller (e.g. stroke) — it benefits from
        the regularisation effect of the other tasks' gradients.
      ✓ You want a single deployable model file.

    It offers no advantage when:
      ✗ You want task-specific hyper-parameters (lr, loss weights).
      ✗ You need per-task ablation numbers (separate probes are cleaner).
    """
    def __init__(self, feature_dim: int, num_classes: int = 2):
        super().__init__()
        self.heads = nn.ModuleDict({
            task: nn.Conv2d(feature_dim, num_classes, kernel_size=1)
            for task in TASK_NAMES
        })
        for head in self.heads.values():
            nn.init.kaiming_normal_(head.weight)

    def forward(self, features, task: str):
        """
        Args:
            features : [B, C, H, W] – SAM3 feature map
            task     : one of 'tumor', 'ms', 'stroke'
        Returns:
            logits   : [B, num_classes, H, W]
        """
        assert task in self.heads, f"Unknown task '{task}'"
        return self.heads[task](features)

    def forward_all(self, features):
        """Run all heads; returns a dict of logits."""
        return {task: head(features) for task, head in self.heads.items()}


# ─────────────────────────────────────────────────────────────────
# LOSS HELPERS
# ─────────────────────────────────────────────────────────────────

def build_criterion(class_weights=None, device='cuda'):
    """
    Weighted CrossEntropyLoss.

    MS and stroke have severe class imbalance (tiny lesions on large background).
    Pass class_weights=[w_bg, w_fg] computed from the dataset to compensate.
    For BraTS tumors the imbalance is less extreme; equal weights often work.
    """
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weight)
    return nn.CrossEntropyLoss()


def dice_loss(logits, targets, smooth=1.0):
    """
    Soft Dice loss – useful addition to CE for highly imbalanced tasks.
    logits : [B, 2, H, W]
    targets: [B, H, W]  (long)
    """
    probs = torch.softmax(logits, dim=1)[:, 1]   # foreground probability
    targets_f = targets.float()
    intersection = (probs * targets_f).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(logits, targets, ce_criterion, dice_weight=0.5):
    """CE + Dice combination; recommended for MS and stroke probes."""
    return ce_criterion(logits, targets) + dice_weight * dice_loss(logits, targets)


# ─────────────────────────────────────────────────────────────────
# TRAINING — ARCHITECTURE A (separate probe, single task)
# ─────────────────────────────────────────────────────────────────

def train_separate_probe(
    train_loader,
    val_loader,
    feature_dim,
    task_name,                  # 'tumor' | 'ms' | 'stroke'
    num_epochs=20,
    lr=1e-3,
    class_weights=None,         # e.g. [1.0, 8.5] for MS/stroke
    use_dice=True,              # recommended for small-lesion tasks
    device='cuda',
):
    """
    Train a single LinearProbe for one task.

    Usage
    -----
        # Tumor (BraTS) – balanced enough for plain CE
        train_separate_probe(..., task_name='tumor', use_dice=False)

        # MS (MSLesSeg) – severe imbalance, use weighted loss + Dice
        weights = ms_dataset.compute_class_weights()
        train_separate_probe(..., task_name='ms',
                             class_weights=weights, use_dice=True)

        # Stroke (ISLES) – same strategy
        weights = stroke_dataset.compute_class_weights()
        train_separate_probe(..., task_name='stroke',
                             class_weights=weights, use_dice=True)
    """
    probe = LinearProbe(feature_dim=feature_dim).to(device)
    criterion = build_criterion(class_weights, device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler()

    # LR scheduler: cosine annealing works well for probes
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    history = {
        'task': task_name,
        'train_losses': [], 'val_accuracies': [], 'val_ious': [], 'val_dices': []
    }
    best_iou = 0.0
    best_state = None

    for epoch in range(num_epochs):
        probe.train()
        epoch_loss = 0.0

        for batch_idx, (features, masks) in enumerate(tqdm(train_loader,
                                    desc=f"[{task_name}] Epoch {epoch+1}/{num_epochs}")):
            features, masks = features.to(device), masks.to(device)
            
            optimizer.zero_grad()

            with autocast():
                logits = probe(features)
                if epoch == 0 and batch_idx == 0:
                    mask_exp = masks.unsqueeze(1).expand_as(features)
                    fg = features[mask_exp == 1]
                    bg = features[mask_exp == 0]

                    print(f"\n[{task_name}] SAM3 feature diagnostic:")

                    if fg.numel() > 0:
                        print(f"  Lesion mean activation : {fg.mean():.4f}")
                        print(f"  BG mean activation     : {bg.mean():.4f}")
                        ratio = fg.mean() / (bg.mean() + 1e-8)
                        print(f"  Ratio fg/bg            : {ratio.item():.4f}")
                    else:
                        print("  ⚠ No lesion pixels in batch")

                    preds_positive = (logits.argmax(dim=1) == 1).float().mean()
                    print(f"  Fraction of positive predictions: {preds_positive:.4f}")
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                if use_dice:
                    loss = combined_loss(logits, masks, criterion)
                else:
                    loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        history['train_losses'].append(avg_loss)

        acc, iou, dice = evaluate_probe(val_loader, probe, device)
        history['val_accuracies'].append(acc)
        history['val_ious'].append(iou)
        history['val_dices'].append(dice)

        if iou > best_iou:
            best_iou = iou
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        print(f"  [{task_name}] Ep {epoch+1:02d}: "
              f"loss={avg_loss:.4f}  acc={acc:.4f}  IoU={iou:.4f}  Dice={dice:.4f}")

    # Restore best weights
    if best_state:
        probe.load_state_dict(best_state)

    history['best_iou'] = best_iou
    return probe, history


# ─────────────────────────────────────────────────────────────────
# TRAINING — ARCHITECTURE B (multi-head, joint training)
# ─────────────────────────────────────────────────────────────────

class TaggedDataset(Dataset):
    """Wraps an HDF5 features dataset and attaches a task label."""
    def __init__(self, base_dataset, task_name):
        self.base = base_dataset
        self.task = task_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        features, masks = self.base[idx]
        return features, masks, self.task


def collate_tagged(batch):
    """Custom collate that preserves task strings."""
    features = torch.stack([b[0] for b in batch])
    masks    = torch.stack([b[1] for b in batch])
    tasks    = [b[2] for b in batch]
    return features, masks, tasks


def train_multihead_probe(
    loaders_dict,               # {'tumor': (train_loader, val_loader), ...}
    feature_dim,
    num_epochs=20,
    lr=1e-3,
    class_weights_dict=None,    # {'tumor': None, 'ms': [1,8.5], 'stroke': [1,6]}
    use_dice=True,
    device='cuda',
):
    """
    Train a MultiHeadProbe by interleaving batches from all three tasks.

    Strategy: each training step samples one batch per task and accumulates
    the gradients before stepping the optimizer. This ensures all heads
    see similar gradient magnitudes regardless of dataset size differences.

    Args:
        loaders_dict : dict mapping task_name → (train_loader, val_loader)
    """
    probe     = MultiHeadProbe(feature_dim=feature_dim).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    # Build one criterion per task (possibly with different class weights)
    criteria = {}
    for task in TASK_NAMES:
        w = (class_weights_dict or {}).get(task, None)
        criteria[task] = build_criterion(w, device)

    history = {task: {'train_losses': [], 'val_ious': [], 'val_dices': []}
               for task in TASK_NAMES}

    for epoch in range(num_epochs):
        probe.train()
        # Make all train loaders into iterators for interleaved sampling
        train_iters = {
            task: iter(loaders_dict[task][0]) for task in TASK_NAMES
        }
        # Use the length of the largest loader as epoch length
        epoch_steps = max(len(loaders_dict[t][0]) for t in TASK_NAMES)

        epoch_losses = {t: 0.0 for t in TASK_NAMES}
        step_counts  = {t: 0   for t in TASK_NAMES}

        for _ in tqdm(range(epoch_steps),
                      desc=f"[multi-head] Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for task in TASK_NAMES:
                try:
                    features, masks = next(train_iters[task])
                except StopIteration:
                    # Restart exhausted loader
                    train_iters[task] = iter(loaders_dict[task][0])
                    features, masks = next(train_iters[task])

                features, masks = features.to(device), masks.to(device)

                with autocast():
                    logits = probe(features, task)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = torch.nn.functional.interpolate(
                            logits, size=masks.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    if use_dice:
                        task_loss = combined_loss(logits, masks, criteria[task])
                    else:
                        task_loss = criteria[task](logits, masks)

                total_loss = total_loss + task_loss
                epoch_losses[task] += task_loss.item()
                step_counts[task]  += 1

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Validation per task
        print(f"\n[multi-head] Epoch {epoch+1} summary:")
        for task in TASK_NAMES:
            avg_loss = epoch_losses[task] / max(step_counts[task], 1)
            history[task]['train_losses'].append(avg_loss)

            _, val_loader = loaders_dict[task]

            def _fwd(feats):
                return probe(feats, task)

            acc, iou, dice = evaluate_probe(val_loader, probe, device,
                                            forward_fn=_fwd)
            history[task]['val_ious'].append(iou)
            history[task]['val_dices'].append(dice)
            print(f"  {task:8s}: loss={avg_loss:.4f}  IoU={iou:.4f}  Dice={dice:.4f}")

    return probe, history


# ─────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate_probe(data_loader, probe, device, forward_fn=None):
    """
    Compute pixel accuracy, IoU, and Dice for binary segmentation.

    forward_fn: optional callable(features) → logits.
                If None, calls probe(features) directly (Architecture A).
    """
    probe.eval()
    tp = fp = tn = fn = 0
    total_pred_pos = 0
    total_gt_pos = 0
    total_pixels = 0

    with torch.no_grad():
        for features, masks in tqdm(data_loader, desc='  Evaluating', leave=False):
            features = features.to(device)
            masks    = masks.to(device)

            logits = forward_fn(features) if forward_fn else probe(features)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )

            preds    = logits.argmax(dim=1)
            preds_fg = preds == 1
            masks_fg = masks == 1

            tp += (preds_fg &  masks_fg).sum().item()
            fp += (preds_fg & ~masks_fg).sum().item()
            fn += (~preds_fg & masks_fg).sum().item()
            tn += (~preds_fg & ~masks_fg).sum().item()

            total_pred_pos += preds_fg.sum().item()    
            total_gt_pos   += masks_fg.sum().item()    
            total_pixels   += masks.numel()    

    total    = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)
    iou      = tp / max(tp + fp + fn, 1)
    dice     = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"  Predicted positive: {100*total_pred_pos/total_pixels:.2f}%  "
          f"| GT positive: {100*total_gt_pos/total_pixels:.2f}%")
    return accuracy, iou, dice


# ─────────────────────────────────────────────────────────────────
# RESULTS PLOTTING
# ─────────────────────────────────────────────────────────────────

def plot_separate_results(histories_dict, save_path):
    """
    histories_dict: {'tumor': history, 'ms': history, 'stroke': history}
    """
    n = len(histories_dict)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (task, h) in enumerate(histories_dict.items()):
        axes[row, 0].plot(h['train_losses'])
        axes[row, 0].set_title(f'{task} – Loss')
        axes[row, 0].set_xlabel('Epoch'); axes[row, 0].grid(True)

        axes[row, 1].plot(h['val_ious'], label='IoU')
        axes[row, 1].plot(h['val_dices'], label='Dice', linestyle='--')
        axes[row, 1].set_title(f'{task} – Val IoU / Dice')
        axes[row, 1].legend(); axes[row, 1].grid(True)

        axes[row, 2].plot(h['val_accuracies'])
        axes[row, 2].set_title(f'{task} – Val Accuracy')
        axes[row, 2].grid(True)

    plt.suptitle('SAM3 Linear Probing – Separate Probes', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Results saved to {save_path}")


def print_comparison_table(histories_dict):
    """Print a summary table comparing all tasks."""
    print("\n" + "=" * 60)
    print(f"{'Task':<10} {'Best IoU':>10} {'Best Dice':>11} {'Final Acc':>10}")
    print("-" * 60)
    for task, h in histories_dict.items():
        best_iou   = max(h.get('val_ious', [0]))
        best_dice  = max(h.get('val_dices', [0]))
        final_acc  = h.get('val_accuracies', [0])[-1] if h.get('val_accuracies') else 0
        print(f"{task:<10} {best_iou:>10.4f} {best_dice:>11.4f} {final_acc:>10.4f}")
    print("=" * 60 + "\n")
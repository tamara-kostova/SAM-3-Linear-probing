#!/usr/bin/env python3
"""
Inference script for trained linear probe

Usage:
    python inference.py --checkpoint ./checkpoints/final_probe.pth \
                        --features_path ./checkpoints/val_features.h5 \
                        --feature_dim 256
"""

import argparse
import torch
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.metrics import accuracy_score, jaccard_score

from models import LinearProbe


def load_model(checkpoint_path, feature_dim=256, num_classes=2, device='cuda'):
    """Load trained linear probe from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Initialize model architecture
    model = LinearProbe(feature_dim=feature_dim, num_classes=num_classes)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def run_inference(model, features_path, device='cuda', batch_size=32):
    """Run inference on pre-computed features"""
    print(f"\nRunning inference on {features_path}...")

    # Load features from HDF5
    with h5py.File(features_path, 'r') as h5f:
        features = h5f['features']
        masks = h5f['masks']
        total_samples = len(features)

        print(f"✓ Loaded {total_samples} samples")
        print(f"  Feature shape: {features[0].shape}")
        print(f"  Mask shape: {masks[0].shape}")

        all_preds = []
        all_targets = []

        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, total_samples, batch_size), desc="Inference"):
                end_idx = min(i + batch_size, total_samples)

                # Load batch
                batch_features = torch.from_numpy(features[i:end_idx]).float().to(device)
                batch_masks = torch.from_numpy(masks[i:end_idx]).long()

                # Forward pass
                logits = model(batch_features)

                # Resize if needed
                if logits.shape[-2:] != batch_masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=batch_masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu().numpy().flatten())
                all_targets.append(batch_masks.numpy().flatten())

    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds, average='binary')

    print(f"\n{'='*60}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"Samples processed: {total_samples}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"IoU (Dice):        {iou:.4f}")
    print(f"{'='*60}")

    return {
        'accuracy': accuracy,
        'iou': iou,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained linear probe')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--features_path', type=str, required=True,
                        help='Path to pre-computed features (.h5)')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension (must match training)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Features:      {args.features_path}")
    print(f"Feature dim:   {args.feature_dim}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Device:        {args.device}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(
        args.checkpoint,
        feature_dim=args.feature_dim,
        device=args.device
    )

    # Run inference
    run_inference(
        model,
        args.features_path,
        device=args.device,
        batch_size=args.batch_size
    )

    print("\n Inference complete!")


if __name__ == '__main__':
    main()

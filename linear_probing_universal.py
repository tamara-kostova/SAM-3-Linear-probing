#!/usr/bin/env python3
"""
Linear Probing Script
Supports: BraTS (tumors), MS (lesions), Stroke (lesions)

This is your existing linear_probing_hdf5.py but modified to support multiple datasets.

Usage:
    # BraTS (you already did this)
    python linear_probe_universal.py \
        --data_root /path/to/BraTS2020 \
        --dataset_type brats \
        --modality t1ce \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --save_dir ./checkpoints_brats
    
    # MS lesion segmentation
    python linear_probe_universal.py \
        --data_root /path/to/MSLesSeg \
        --dataset_type ms \
        --modality flair \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --save_dir ./checkpoints_ms
    
    # Stroke lesion segmentation
    python linear_probe_universal.py \
        --data_root /path/to/ISLES2022 \
        --dataset_type stroke \
        --modality dwi \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --save_dir ./checkpoints_stroke
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Import the dataset class
from dataset import MedicalDataset


class LinearProbe(nn.Module):
    """Simple linear classifier for segmentation"""
    def __init__(self, feature_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    
    def forward(self, features):
        logits = self.classifier(features)
        return logits


class SAM3FeatureExtractor:
    """Extracts features from frozen SAM3 encoder"""
    def __init__(self, sam3_model, device='cuda'):
        self.model = sam3_model
        self.device = device
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        self.model.to(device)
        print("✓ SAM3 backbone frozen")
    
    @torch.no_grad()
    def extract_features(self, images, captions=None):
        if captions is None:
            batch_size = images.shape[0]
            captions = ["lesion"] * batch_size  # Generic caption
        
        backbone_out = self.model.backbone(images, captions)
        
        # Extract features from dict output
        if isinstance(backbone_out, dict):
            for key in ['vision_features', 'image_features', 'features']:
                if key in backbone_out:
                    return backbone_out[key]
            # Fallback: first 4D tensor
            for v in backbone_out.values():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    return v
        
        return backbone_out


def precompute_features_hdf5(dataset, feature_extractor, save_path, 
                              batch_size=16, device='cuda'):
    """
    Pre-compute SAM3 features and save to HDF5 (memory-efficient).
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    print(f"\n{'='*80}")
    print("PRE-COMPUTING SAM3 FEATURES")
    print('='*80)
    print(f"Batch size: {batch_size}")
    print(f"Total samples: {len(dataset)}")
    print(f"Save path: {save_path}")
    print()
    
    # Test first batch
    print("Testing feature extraction...")
    first_images, first_masks = next(iter(dataloader))
    first_images = first_images.to(device)
    
    with torch.no_grad():
        test_features = feature_extractor.extract_features(first_images)
        if test_features is None:
            raise ValueError("Feature extractor returned None!")
    
    feature_shape = test_features.shape[1:]  # [C, H, W]
    mask_shape = first_masks.shape[1:]
    total_samples = len(dataset)
    
    print(f"✓ Test passed!")
    print(f"  Feature shape: {feature_shape}")
    print(f"  Mask shape: {mask_shape}")
    print()
    
    # Create HDF5 file
    print(f"Creating HDF5 file...")
    with h5py.File(save_path, 'w') as h5f:
        # Create datasets
        features_dset = h5f.create_dataset(
            'features',
            shape=(total_samples, *feature_shape),
            dtype='float32',
            chunks=(1, *feature_shape),
            compression='gzip',
            compression_opts=4
        )
        
        masks_dset = h5f.create_dataset(
            'masks',
            shape=(total_samples, *mask_shape),
            dtype='int64',
            chunks=(1, *mask_shape),
            compression='gzip',
            compression_opts=4
        )
        
        # Store metadata
        h5f.attrs['feature_dim'] = feature_shape[0]
        h5f.attrs['total_samples'] = total_samples
        
        # Process batches
        current_idx = 0
        
        for images, masks in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            with torch.no_grad():
                features = feature_extractor.extract_features(images)
                
                # Write to disk
                features_dset[current_idx:current_idx + batch_size_actual] = features.cpu().numpy()
                masks_dset[current_idx:current_idx + batch_size_actual] = masks.numpy()
                
                current_idx += batch_size_actual
                
                # Free memory
                del features, images
                if current_idx % 160 == 0:  # Every 10 batches
                    torch.cuda.empty_cache()
        
        print(f"\n✓ Saved {current_idx} samples")
    
    file_size_mb = os.path.getsize(save_path) / 1e6
    print(f"✓ File size: {file_size_mb:.1f} MB\n")


class HDF5FeaturesDataset(torch.utils.data.Dataset):
    """Dataset that streams features from HDF5 file"""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        
        with h5py.File(h5_path, 'r') as h5f:
            self.total_samples = h5f.attrs['total_samples']
            self.feature_dim = h5f.attrs['feature_dim']
        
        print(f"✓ Opened {h5_path}")
        print(f"  Samples: {self.total_samples}")
        print(f"  Feature dim: {self.feature_dim}")
        
        self.h5f = None
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
        
        features = torch.from_numpy(self.h5f['features'][idx]).float()
        masks = torch.from_numpy(self.h5f['masks'][idx]).long()
        
        return features, masks


def train_linear_probe(train_loader, val_loader, probe, num_epochs, lr, device):
    """Train linear probe"""
    probe.to(device)
    
    try:
        probe = torch.compile(probe)
        print("✓ Model compiled")
    except:
        pass
    
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    history = {'train_losses': [], 'val_accuracies': [], 'val_ious': []}
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training
        probe.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for features, masks in pbar:
            features = features.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = probe(features)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_losses'].append(avg_loss)
        
        # Validation
        val_acc, val_iou = evaluate_probe(val_loader, probe, device)
        history['val_accuracies'].append(val_acc)
        history['val_ious'].append(val_iou)
        
        if val_iou > best_iou:
            best_iou = val_iou
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val IoU={val_iou:.4f}")
    
    history['best_iou'] = best_iou
    return history


def evaluate_probe(data_loader, probe, device):
    """Evaluate probe"""
    probe.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, masks in tqdm(data_loader, desc="Evaluating", leave=False):
            features = features.to(device)
            masks = masks.to(device)
            
            logits = probe(features)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds, average='binary', zero_division=0)
    
    return accuracy, iou


def plot_results(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_losses'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(history['val_accuracies'])
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    axes[2].plot(history['val_ious'])
    axes[2].set_title('Validation IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    """Main pipeline"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Paths
    cache_dir = os.path.join(args.save_dir, "cache")
    features_train_path = os.path.join(args.save_dir, "train_features.h5")
    features_val_path = os.path.join(args.save_dir, "val_features.h5")
    
    print(f"\n{'='*80}")
    print(f"LINEAR PROBING - {args.dataset_type.upper()}")
    print('='*80)
    print(f"Dataset type:  {args.dataset_type}")
    print(f"Data root:     {args.data_root}")
    print(f"Modality:      {args.modality}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Device:        {device}")
    print('='*80 + '\n')
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = MedicalDataset(
        data_root=args.data_root,
        dataset_type=args.dataset_type,
        modality=args.modality,
        cache_dir=cache_dir,
        force_rebuild=args.rebuild_cache
    )
    
    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Train: {len(train_dataset)} slices")
    print(f"✓ Val: {len(val_dataset)} slices")
    
    # Pre-compute features if needed
    if not os.path.exists(features_train_path):
        from sam3.sam3 import build_sam3_image_model
        
        print("\nLoading SAM3 model...")
        sam3_model = build_sam3_image_model(bpe_path=args.bpe_path)
        feature_extractor = SAM3FeatureExtractor(sam3_model, device=device)
        
        # Pre-compute features
        precompute_features_hdf5(
            train_dataset, feature_extractor,
            features_train_path, batch_size=16, device=device
        )
        precompute_features_hdf5(
            val_dataset, feature_extractor,
            features_val_path, batch_size=16, device=device
        )
        
        print("✓ Features pre-computed!")
        
        # Free memory
        del sam3_model, feature_extractor
        torch.cuda.empty_cache()
    else:
        print("\n✓ Using existing pre-computed features")
    
    # Load features
    print(f"\n{'='*80}")
    print("LOADING PRE-COMPUTED FEATURES")
    print('='*80)
    
    train_features_dataset = HDF5FeaturesDataset(features_train_path)
    val_features_dataset = HDF5FeaturesDataset(features_val_path)
    
    train_loader = DataLoader(
        train_features_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_features_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize probe
    feature_dim = train_features_dataset.feature_dim
    probe = LinearProbe(feature_dim=feature_dim, num_classes=2)
    
    print(f"\n✓ Linear probe initialized")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Parameters: {sum(p.numel() for p in probe.parameters()):,}")
    
    # Train
    print(f"\n{'='*80}")
    print("TRAINING")
    print('='*80 + '\n')
    
    history = train_linear_probe(
        train_loader, val_loader, probe,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Save results
    torch.save({
        'model_state_dict': probe.state_dict(),
        'feature_dim': feature_dim,
        'history': history,
        'dataset_type': args.dataset_type,
        'modality': args.modality
    }, os.path.join(args.save_dir, 'final_probe.pth'))
    
    plot_results(history, os.path.join(args.save_dir, 'results.png'))
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print('='*80)
    print(f"Best validation IoU: {history['best_iou']:.4f}")
    print(f"Final validation IoU: {history['val_ious'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    print('='*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Linear Probing for Medical Image Segmentation'
    )
    
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['brats', 'ms', 'stroke'])
    parser.add_argument('--modality', type=str, required=True,
                        help='Image modality (flair, t1ce, dwi, etc.)')
    parser.add_argument('--bpe_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rebuild_cache', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
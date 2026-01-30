#!/usr/bin/env python3
"""
SAM3 Linear Probing - Fixed Feature Extraction

This version includes:
- Better error handling for feature extraction
- Debugging information
- Fallback strategies
- Progress tracking
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import h5py

from models import LinearProbe

# ============================================================================
# 1. DATASET
# ============================================================================

class PreprocessedBraTSDataset(Dataset):
    """
    Pre-processes and caches all slices to disk for fast loading.
    Run once with force_rebuild=True, then use force_rebuild=False.
    """
    def __init__(self, data_root, cache_dir, modality='t1ce', 
                 slice_range=(50, 130), img_size=1008, force_rebuild=False):
        """
        Args:
            data_root: Path to BraTS data
            cache_dir: Where to save pre-processed files
            force_rebuild: If True, rebuild cache even if it exists
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_index_file = os.path.join(cache_dir, 'slice_paths.txt')
        
        if not force_rebuild and os.path.exists(cache_index_file):
            # Load existing cache
            print("Loading from existing cache...")
            with open(cache_index_file, 'r') as f:
                self.cache_paths = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(self.cache_paths)} cached slices")
        else:
            # Build cache
            print("Building cache (this takes 30-60 min but only once)...")
            self.cache_paths = self._preprocess_and_cache(
                data_root, modality, slice_range, img_size
            )
            # Save cache index
            with open(cache_index_file, 'w') as f:
                f.write('\n'.join(self.cache_paths))
            print(f"✓ Cache built: {len(self.cache_paths)} slices")
    
    def _preprocess_and_cache(self, data_root, modality, slice_range, img_size):
        """Pre-process all slices once and save to disk"""
        patient_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS20_Training_*")))
        cache_paths = []
        
        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            patient_id = os.path.basename(patient_dir)
            img_path = os.path.join(patient_dir, f"{patient_id}_{modality}.nii")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")
            
            # Handle non-standard naming (e.g., Patient 355)
            if not os.path.exists(img_path):
                alt_files = glob.glob(os.path.join(patient_dir, f"*{modality}*.nii"))
                if alt_files:
                    img_path = alt_files[0]
            
            if not os.path.exists(seg_path):
                alt_seg = glob.glob(os.path.join(patient_dir, "*[Ss]eg*.nii"))
                if alt_seg:
                    seg_path = alt_seg[0]
            
            if not (os.path.exists(img_path) and os.path.exists(seg_path)):
                print(f"  Warning: Skipping {patient_id} - missing files")
                continue
            
            # Load volumes ONCE per patient
            img_data = nib.load(img_path).get_fdata()
            seg_data = nib.load(seg_path).get_fdata()
            
            # Process all slices for this patient
            for slice_idx in range(slice_range[0], slice_range[1]):
                img_slice = img_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]
                
                # Normalize
                img_min, img_max = img_slice.min(), img_slice.max()
                if img_max > img_min:
                    img_slice = (img_slice - img_min) / (img_max - img_min)
                
                # Convert to RGB
                img_slice_rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
                img_tensor = torch.from_numpy(img_slice_rgb).float().permute(2, 0, 1)
                
                # Binary segmentation
                seg_binary = (seg_slice > 0).astype(np.float32)
                seg_tensor = torch.from_numpy(seg_binary).float()
                
                # Resize
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=(img_size, img_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                
                seg_tensor = torch.nn.functional.interpolate(
                    seg_tensor.unsqueeze(0).unsqueeze(0), size=(img_size, img_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
                
                # Save to cache
                cache_path = os.path.join(self.cache_dir, f"{patient_id}_{slice_idx:03d}.pt")
                torch.save({'image': img_tensor, 'mask': seg_tensor}, cache_path)
                cache_paths.append(cache_path)
        
        return cache_paths
    
    def __len__(self):
        return len(self.cache_paths)
    
    def __getitem__(self, idx):
        # Simply load pre-processed tensor - FAST!
        data = torch.load(self.cache_paths[idx])
        return data['image'], data['mask']


def precompute_sam3_features(dataset, feature_extractor, save_path,
                                       batch_size=16, device='cuda'):
    """
    MEMORY-EFFICIENT VERSION: Streams features to HDF5 file and keeps them there.

    This fixes the RAM overflow issue by writing directly to disk and never loading
    the full dataset into RAM.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Reduced to save RAM
        pin_memory=False  # Disabled to save RAM
    )

    print(f"\nPre-computing SAM3 features (STREAMING MODE)...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(dataloader)}")
    print(f"  Save path: {save_path}")
    print(f"  Using HDF5 format - features stay on disk")
    print()

    # Test first batch to get dimensions
    print("Testing feature extraction on first batch...")
    first_images, first_masks = next(iter(dataloader))
    first_images = first_images.to(device)

    with torch.no_grad():
        test_features = feature_extractor.extract_features(first_images)
        if test_features is None:
            raise ValueError("Feature extractor returned None!")

    # Get dimensions
    feature_shape = test_features.shape[1:]  # [C, H, W]
    mask_shape = first_masks.shape[1:]  # [H, W]
    total_samples = len(dataset)

    print(f"✓ Feature extraction test passed!")
    print(f"  Input shape: {first_images.shape}")
    print(f"  Feature shape per sample: {feature_shape}")
    print(f"  Mask shape per sample: {mask_shape}")
    print(f"  Total samples to process: {total_samples}")
    print()

    # Create HDF5 file
    print(f"Creating HDF5 file: {save_path}")

    with h5py.File(save_path, 'w') as h5f:
        # Create datasets with compression
        features_dset = h5f.create_dataset(
            'features',
            shape=(total_samples, *feature_shape),
            dtype='float32',
            chunks=(1, *feature_shape),  # One sample per chunk
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

        # Process batches and write directly to disk
        current_idx = 0

        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = images.to(device)
            batch_size_actual = images.shape[0]

            try:
                with torch.no_grad():
                    # Extract features
                    features = feature_extractor.extract_features(images)

                    if features is None:
                        raise ValueError(f"Feature extractor returned None for batch {batch_idx}")

                    # Move to CPU and convert to numpy immediately (free GPU memory)
                    features_np = features.cpu().numpy()
                    masks_np = masks.numpy()

                    # Write to HDF5 (disk)
                    features_dset[current_idx:current_idx + batch_size_actual] = features_np
                    masks_dset[current_idx:current_idx + batch_size_actual] = masks_np

                    current_idx += batch_size_actual

                    # Explicitly delete to free memory
                    del features, features_np, masks_np, images

                    # Clear GPU cache every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ Error in batch {batch_idx}: {e}")
                raise

        print(f"\n✓ Wrote {current_idx} samples to HDF5 file")

    file_size_mb = os.path.getsize(save_path) / 1e6
    print(f"✓ Saved {total_samples} feature vectors to {save_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Feature shape: {feature_shape}")
    print()



class PrecomputedFeaturesDataset(Dataset):
    """Dataset that loads pre-computed SAM3 features from HDF5 file (memory-efficient)"""
    def __init__(self, features_path):
        print(f"Loading pre-computed features from {features_path}...")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        # Open HDF5 file in read mode and keep it open
        self.h5_file = h5py.File(features_path, 'r')
        self.features = self.h5_file['features']
        self.masks = self.h5_file['masks']

        print(f"✓ Loaded {len(self.features)} samples (streaming from disk)")
        print(f"  Feature shape: {self.features[0].shape}")
        print(f"  Mask shape: {self.masks[0].shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load single sample from disk (fast with chunking + compression)
        features = torch.from_numpy(self.features[idx]).float()
        masks = torch.from_numpy(self.masks[idx]).long()
        return features, masks

    def __del__(self):
        # Close HDF5 file when dataset is destroyed
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


# ============================================================================
# 2. SAM3 FEATURE EXTRACTOR
# ============================================================================

class SAM3FeatureExtractor:
    """
    Extracts features from frozen SAM3 encoder.
    IMPROVED with better error handling.
    """
    def __init__(self, sam3_model, device='cuda'):
        self.model = sam3_model
        self.device = device

        # Freeze parameters
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for name, p in sam3_model.named_parameters():
                p.requires_grad = False
            for name, p in sam3_model.named_parameters():
                if name.startswith("segmentation_head"):
                    p.requires_grad = True
            print("✓ Froze SAM3 backbone parameters")
        else:
            print("⚠ Warning: SAM3 model doesn't have 'backbone' attribute")
            print(f"  Model type: {type(sam3_model)}")
            print(f"  Available attributes: {dir(sam3_model)[:10]}...")

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def extract_features(self, images, captions=None):
        """
        Extract features from SAM3 backbone.
        
        Args:
            images: [B, 3, H, W] input images
            captions: List of text captions (SAM3 is vision-language model)
        Returns:
            features: Tensor of visual features [B, C, H, W]
        """
        # SAM3 requires captions
        if captions is None:
            batch_size = images.shape[0]
            captions = ["brain tumor"] * batch_size

        try:
            # SAM3 backbone returns a dictionary of features
            backbone_out = self.model.backbone(images, captions)
        except Exception as e:
            print(f"\n❌ Error calling model.backbone:")
            print(f"   {e}")
            print(f"   Images shape: {images.shape}")
            print(f"   Captions: {captions[:2]}...")
            raise

        # Extract the visual features from the dictionary
        if isinstance(backbone_out, dict):
            print(f"  Backbone output keys: {list(backbone_out.keys())}")
            
            # Try common keys
            possible_keys = ['vision_features', 'image_features', 'visual_features', 
                           'features', 'vision_embedding', 'image_embedding']
            
            for key in possible_keys:
                if key in backbone_out:
                    features = backbone_out[key]
                    if isinstance(features, torch.Tensor) and len(features.shape) == 4:
                        print(f"  Using '{key}': {features.shape}")
                        return features
            
            # Fallback: return first 4D tensor
            for key, val in backbone_out.items():
                if isinstance(val, torch.Tensor) and len(val.shape) == 4:
                    print(f"  Using '{key}' (fallback): {val.shape}")
                    return val
            
            # If we get here, no suitable features found
            print(f"❌ No 4D tensor found in backbone output!")
            for key, val in backbone_out.items():
                if isinstance(val, torch.Tensor):
                    print(f"   {key}: {val.shape}")
            raise ValueError("Could not find feature tensor in SAM3 output")
        
        elif isinstance(backbone_out, torch.Tensor):
            print(f"  Backbone output shape: {backbone_out.shape}")
            return backbone_out
        
        else:
            raise ValueError(f"Unexpected backbone output type: {type(backbone_out)}")


# ============================================================================
# 3. TRAINING FUNCTION
# ============================================================================

def train_linear_probe_optimized(
    train_loader,
    val_loader,
    probe,
    num_epochs=20,
    lr=0.001,
    device='cuda',
    use_amp=True,
    accumulation_steps=1
):
    """
    Optimized training with mixed precision and gradient accumulation.
    """
    probe.to(device)
    
    # Try to compile for extra speedup (PyTorch 2.0+)
    try:
        probe = torch.compile(probe)
        print("✓ Model compiled with torch.compile")
    except:
        print("⚠ torch.compile not available, continuing without it")
    
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None
    
    train_losses = []
    val_accuracies = []
    val_ious = []
    
    for epoch in range(num_epochs):
        # Training
        probe.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (features, masks) in enumerate(pbar):
            features = features.to(device)
            masks = masks.to(device)
            
            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    logits = probe(features)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = torch.nn.functional.interpolate(
                            logits, size=masks.shape[-2:], 
                            mode='bilinear', align_corners=False
                        )
                    loss = criterion(logits, masks) / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = probe(features)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                loss = criterion(logits, masks) / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        val_acc, val_iou = evaluate_probe_optimized(val_loader, probe, device)
        val_accuracies.append(val_acc)
        val_ious.append(val_iou)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val IoU={val_iou:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_ious': val_ious
    }


# ============================================================================
# 4. EVALUATION FUNCTION
# ============================================================================

def evaluate_probe_optimized(data_loader, probe, device='cuda'):
    probe.eval()

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for features, masks in tqdm(data_loader, desc="Evaluating", leave=False):
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = probe(features)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )

            preds = torch.argmax(logits, dim=1)  # [B,H,W]

            preds_fg = (preds == 1)
            masks_fg = (masks == 1)

            tp += (preds_fg & masks_fg).sum().item()
            fp += (preds_fg & ~masks_fg).sum().item()
            fn += (~preds_fg & masks_fg).sum().item()
            tn += (~preds_fg & ~masks_fg).sum().item()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)
    iou = tp / max(tp + fp + fn, 1)

    return accuracy, iou


# ============================================================================
# 5. PLOTTING
# ============================================================================

def plot_results(history, save_path='linear_probe_results.png'):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_losses'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['val_accuracies'])
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    # IoU
    axes[2].plot(history['val_ious'])
    axes[2].set_title('Validation IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Results saved to {save_path}")
    plt.close()


# ============================================================================
# 6. MAIN FUNCTION
# ============================================================================

def main(args):
    """Main training pipeline"""
    
    # Configuration from command-line arguments
    data_root = args.data_root
    cache_dir = os.path.join(args.save_dir, "preprocessed_cache")
    features_train_path = os.path.join(args.save_dir, "train_features.h5")
    features_val_path = os.path.join(args.save_dir, "val_features.h5")
    bpe_path = args.bpe_path

    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Data root:     {data_root}")
    print(f"Cache dir:     {cache_dir}")
    print(f"BPE path:      {bpe_path}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Save dir:      {args.save_dir}")
    print("="*80 + "\n")
    
    # Load dataset
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    full_dataset = PreprocessedBraTSDataset(
        data_root=data_root,
        cache_dir=cache_dir,
        modality=args.modality,
        force_rebuild=False
    )
    if len(full_dataset) == 0:
        raise RuntimeError(
            "Dataset is empty after preprocessing. "
            "Check slice_range and cache."
        )

    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )    
    
    print(f"✓ Train: {len(train_dataset)} slices")
    print(f"✓ Val: {len(val_dataset)} slices")

    # Pre-compute features if needed
    if not os.path.exists(features_train_path):
        print("\n" + "="*80)
        print("PRE-COMPUTING SAM3 FEATURES (ONE-TIME ONLY)")
        print("="*80)
        
        from sam3.sam3 import build_sam3_image_model
        
        print(f"\nLoading SAM3 model from: {bpe_path}")
        sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        
        print("Creating feature extractor...")
        feature_extractor = SAM3FeatureExtractor(sam3_model, device=device)
        
        # Pre-compute features
        precompute_sam3_features(
            train_dataset, feature_extractor, 
            features_train_path, batch_size=2, device=device
        )
        precompute_sam3_features(
            val_dataset, feature_extractor,
            features_val_path, batch_size=2, device=device
        )
        
        print("✓ Features pre-computed and saved!")
    else:
        print("\n✓ Using existing pre-computed features")

    # Load pre-computed feature datasets
    print("\n" + "="*80)
    print("LOADING PRE-COMPUTED FEATURES")
    print("="*80)
    
    train_features_dataset = PrecomputedFeaturesDataset(features_train_path)
    val_features_dataset = PrecomputedFeaturesDataset(features_val_path)

    # Data loaders
    train_loader = DataLoader(
        train_features_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_features_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    # Initialize probe
    # Get feature dimension from first batch
    sample_features, _ = train_features_dataset[0]
    feature_dim = sample_features.shape[0]
    print(f"\n✓ Detected feature dimension: {feature_dim}")
    
    probe = LinearProbe(feature_dim=feature_dim, num_classes=2)
    
    probe_params = sum(p.numel() for p in probe.parameters())
    print(f"✓ Linear probe initialized")
    print(f"  Parameters: {probe_params:,}")
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    history = train_linear_probe_optimized(
        train_loader=train_loader,
        val_loader=val_loader,
        probe=probe,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        use_amp=True,
        accumulation_steps=1
    )
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
        
    # Save final model
    model_save_path = os.path.join(args.save_dir, 'final_probe.pth')
    torch.save(probe.state_dict(), model_save_path)
    print(f"✓ Final model saved to {model_save_path}")
    
    # Plot results
    plot_save_path = os.path.join(args.save_dir, 'linear_probe_results.png')
    plot_results(history, save_path=plot_save_path)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation IoU: {max(history['val_ious']):.4f}")
    print(f"Final validation IoU: {history['val_ious'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    print("="*80)


# ============================================================================
# 7. COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train linear probe on frozen SAM3 features for brain tumor segmentation'
    )
    
    # Paths
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to BraTS2020 training data directory')
    parser.add_argument('--bpe_path', type=str, required=True,
                        help='Path to BPE vocabulary file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints and results')
    
    # Dataset parameters
    parser.add_argument('--modality', type=str, default='t1ce',
                        choices=['flair', 't1', 't1ce', 't2'],
                        help='MRI modality to use')
    parser.add_argument('--img_size', type=int, default=1008,
                        help='Image size (SAM3 expects 1008)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
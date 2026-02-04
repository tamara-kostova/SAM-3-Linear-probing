#!/usr/bin/env python3
"""
Test on 125 BraTS 2021 Patients

Simple script to test trained BraTS 2020 model on BraTS 2021 data.

Usage:
    python test_tumor_script.py \
        --test_data /path/to/BraTS2021_Training_Data \
        --checkpoint ./checkpoints/final_probe.pth \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --num_patients 125
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import h5py
import glob
import random
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, jaccard_score, precision_recall_fscore_support
import nibabel as nib
import gc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(results, save_dir):
    metric_keys = ['accuracy', 'iou', 'dice', 'precision', 'recall', 'f1']
    names = metric_keys
    values = [results[k] for k in metric_keys]

    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("BraTS 2021 Test Metrics")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics.png"))
    plt.close()


def plot_predictions(h5_path, model, save_dir, device='cuda', num_samples=5):
    dataset = HDF5Dataset(h5_path)
    model.eval().to(device)

    idxs = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(idxs):
            features, mask = dataset[idx]
            features = features.unsqueeze(0).to(device)

            logits = model(features)
            pred = torch.argmax(logits, dim=1).squeeze().cpu()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("GT Mask")
            plt.imshow(mask, cmap='gray')

            plt.subplot(1, 3, 2)
            plt.title("Prediction")
            plt.imshow(pred, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(mask, cmap='gray')
            plt.imshow(pred, alpha=0.5, cmap='jet')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
            plt.close()

def create_plots(results, save_dir, num_patients):
    """Generate result visualizations"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Background', 'Tumor'],
                yticklabels=['Background', 'Tumor'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # 2. Metrics Bar Chart
    metrics = {
        'Accuracy': results['accuracy'],
        'IoU': results['iou'],
        'Dice': results['dice'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1']
    }
    
    colors = ['#1f77b4' if v >= 0.6 else '#ff7f0e' for v in metrics.values()]
    bars = axes[1].barh(list(metrics.keys()), list(metrics.values()), color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Score')
    axes[1].set_title(f'Test Metrics ({num_patients} patients)')
    axes[1].axvline(x=0.6, color='red', linestyle='--', alpha=0.3, label='0.6 threshold')
    axes[1].legend()
    
    # Add value labels
    for i, (metric, value) in enumerate(metrics.items()):
        axes[1].text(value + 0.02, i, f'{value:.3f}', va='center')
    
    # 3. Class Distribution
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    bg_pixels = tn + fp
    tumor_pixels = tp + fn

    
    axes[2].pie([bg_pixels, tumor_pixels], 
                labels=['Background', 'Tumor'],
                autopct='%1.1f%%',
                colors=['#1f77b4', '#ff7f0e'])
    axes[2].set_title('Class Distribution')
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'test_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to {plot_path}")
    plt.close()

class LinearProbe(nn.Module):
    def __init__(self, feature_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    
    def forward(self, features):
        return self.classifier(features)


class SAM3FeatureExtractor:
    def __init__(self, sam3_model, device='cuda'):
        self.model = sam3_model
        self.device = device
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(device)
    
    @torch.no_grad()
    def extract_features(self, images, captions=None):
        if captions is None:
            captions = ["brain tumor"] * images.shape[0]
        backbone_out = self.model.backbone(images, captions)
        if isinstance(backbone_out, dict):
            for key in ['vision_features', 'image_features', 'features']:
                if key in backbone_out:
                    return backbone_out[key]
            for v in backbone_out.values():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    return v
        return backbone_out


def select_random_patients(data_root, num_patients=125, seed=42):
    all_patients = sorted(glob.glob(os.path.join(data_root, "BraTS2021_*")))
    print(f"\n{'='*80}")
    print("PATIENT SELECTION")
    print('='*80)
    print(f"Total: {len(all_patients)} patients")
    print(f"Selecting: {num_patients} patients")
    
    random.seed(seed)
    selected = random.sample(all_patients, min(num_patients, len(all_patients)))
    
    print(f"✓ Selected {len(selected)} patients (seed={seed})")
    print(f"\nFirst 5 selected:")
    for i, p in enumerate(selected[:5]):
        print(f"  {i+1}. {os.path.basename(p)}")
    print('='*80)
    return selected


def create_efficient_cache(selected_patients, cache_dir, modality='t1ce',
                           slice_range=(50, 130), img_size=1008):
    """
    Efficient caching using float16 properly
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_paths = []
    
    # Check if cache already exists
    existing_cache = list(Path(cache_dir).glob("*.pt"))
    if len(existing_cache) > 0:
        print(f"\n{'='*80}")
        print("USING EXISTING CACHE")
        print('='*80)
        print(f"Found {len(existing_cache)} cached slices")
        print(f"Cache directory: {cache_dir}")
        
        # Return existing cache paths
        cache_paths = [str(f) for f in sorted(existing_cache)]
        
        cache_size_gb = sum(f.stat().st_size for f in existing_cache) / 1e9
        print(f"✓ Cache size: {cache_size_gb:.2f} GB")
        print('='*80)
        
        return cache_paths
    
    print(f"\n{'='*80}")
    print("CACHING PATIENTS (MEMORY-EFFICIENT)")
    print('='*80)
    print(f"Patients: {len(selected_patients)}")
    print(f"Expected cache: ~6 GB (using float16)")
    
    for patient_dir in tqdm(selected_patients, desc="Caching"):
        patient_id = os.path.basename(patient_dir)
        
        # Find files
        img_files = glob.glob(os.path.join(patient_dir, f"*{modality}*.nii*"))
        seg_files = glob.glob(os.path.join(patient_dir, "*seg*.nii*"))
        
        if not img_files or not seg_files:
            continue
        
        try:
            img_data = nib.load(img_files[0]).get_fdata()
            seg_data = nib.load(seg_files[0]).get_fdata()
            
            for slice_idx in range(slice_range[0], min(slice_range[1], img_data.shape[2])):
                img_slice = img_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]
                
                # Normalize
                img_min, img_max = img_slice.min(), img_slice.max()
                if img_max > img_min:
                    img_slice = (img_slice - img_min) / (img_max - img_min)
                
                # Convert to float16 BEFORE creating tensor
                img_slice_rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
                img_slice_rgb = img_slice_rgb.astype(np.float16)  # Convert numpy array to float16!
                
                # Now create tensor (already float16)
                img_tensor = torch.from_numpy(img_slice_rgb).permute(2, 0, 1)
                
                # Resize (convert to float32 temporarily for interpolation, then back)
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0).float(), 
                    size=(img_size, img_size),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).half()  # Convert back to float16!
                
                # Mask: uint8 for efficiency
                seg_binary = (seg_slice > 0).astype(np.uint8)
                seg_tensor = torch.from_numpy(seg_binary)
                
                seg_tensor = torch.nn.functional.interpolate(
                    seg_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0).byte()
                
                # Save with compression
                cache_path = os.path.join(cache_dir, f"{patient_id}_{slice_idx:03d}.pt")
                torch.save(
                    {'image': img_tensor, 'mask': seg_tensor},
                    cache_path,
                    _use_new_zipfile_serialization=True
                )
                cache_paths.append(cache_path)
        
        except Exception as e:
            print(f"\n⚠ Error: {patient_id}: {e}")
            continue
    
    # Check actual size
    cache_size_gb = sum(f.stat().st_size for f in Path(cache_dir).glob("*.pt")) / 1e9
    
    print(f"\n✓ Cached {len(cache_paths)} slices")
    print(f"✓ Cache size: {cache_size_gb:.2f} GB")
    
    if cache_size_gb > 10:
        print(f"⚠ WARNING: Cache is {cache_size_gb:.1f} GB (expected ~6 GB)")
        print("  Something may be wrong with float16 conversion!")
    else:
        print("✓ Cache size is efficient!")
    
    print('='*80)
    return cache_paths


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths):
        self.cache_paths = cache_paths
    
    def __len__(self):
        return len(self.cache_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.cache_paths[idx])
        return data['image'].float(), data['mask'].long()


def precompute_features(dataset, feature_extractor, save_path, batch_size=16, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"\n{'='*80}")
    print("EXTRACTING SAM3 FEATURES")
    print('='*80)
    
    first_images, first_masks = next(iter(dataloader))
    with torch.no_grad():
        test_features = feature_extractor.extract_features(first_images.to(device))
    
    feature_shape = test_features.shape[1:]
    total_samples = len(dataset)
    
    print(f"Samples: {total_samples}")
    print(f"Feature shape: {feature_shape}")
    
    with h5py.File(save_path, 'w') as h5f:
        features_dset = h5f.create_dataset(
            'features', shape=(total_samples, *feature_shape),
            dtype='float32', chunks=(1, *feature_shape),
            compression='gzip', compression_opts=4
        )
        masks_dset = h5f.create_dataset(
            'masks', shape=(total_samples, *first_masks.shape[1:]),
            dtype='int64', chunks=(1, *first_masks.shape[1:]),
            compression='gzip', compression_opts=4
        )
        h5f.attrs['feature_dim'] = feature_shape[0]
        h5f.attrs['total_samples'] = total_samples
        
        current_idx = 0
        for images, masks in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            with torch.no_grad():
                features = feature_extractor.extract_features(images)
                features_dset[current_idx:current_idx + len(images)] = features.cpu().numpy()
                masks_dset[current_idx:current_idx + len(images)] = masks.numpy()
                current_idx += len(images)
                del features, images
                if current_idx % 160 == 0:
                    torch.cuda.empty_cache()
    
    print(f"✓ Saved {current_idx} features ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print('='*80)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as h5f:
            self.total_samples = h5f.attrs['total_samples']
            self.feature_dim = h5f.attrs['feature_dim']
        self.h5f = None
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
        return (torch.from_numpy(self.h5f['features'][idx]).float(),
                torch.from_numpy(self.h5f['masks'][idx]).long())


def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print('='*80)
    
    eval_size = 256  # Downsample for evaluation to save memory
    tp = fp = fn = tn = 0

    print(f"\n{'='*80}")
    print("RUNNING INFERENCE (STREAMING METRICS)")
    print('='*80)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        for features, masks in tqdm(test_loader, desc="Testing"):
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(features)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            preds = torch.argmax(logits, dim=1)

            # Move to CPU once
            preds = preds.cpu().numpy().astype(np.uint8).ravel()
            masks = masks.cpu().numpy().astype(np.uint8).ravel()

            tp += np.sum((preds == 1) & (masks == 1))
            fp += np.sum((preds == 1) & (masks == 0))
            fn += np.sum((preds == 0) & (masks == 1))
            tn += np.sum((preds == 0) & (masks == 0))

            del features, masks, logits, preds
            torch.cuda.empty_cache()

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    cm = np.array([[tn, fp],
                   [fn, tp]])

    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print('='*80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print('='*80)

    return {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TESTING ON {args.num_patients} BRATS 2021 PATIENTS")
    print('='*80)
    print(f"Test data: {args.test_data}")
    print(f"Device: {device}")
    print('='*80)
    
    # Select patients
    selected = select_random_patients(args.test_data, args.num_patients, args.seed)
    
    with open(os.path.join(args.save_dir, 'patients.txt'), 'w') as f:
        for p in selected:
            f.write(os.path.basename(p) + '\n')
    
    # Cache
    cache_paths = create_efficient_cache(selected, os.path.join(args.save_dir, "cache"))
    dataset = CachedDataset(cache_paths)
    
    features_path = os.path.join(args.save_dir, "features.h5")
    # Extract features
    if not os.path.exists(features_path):
        from sam3.sam3 import build_sam3_image_model
        print("\nLoading SAM3...")
        sam3 = build_sam3_image_model(bpe_path=args.bpe_path)
        extractor = SAM3FeatureExtractor(sam3, device)
        
        precompute_features(dataset, extractor, features_path, 16, device)
        del sam3, extractor
    torch.cuda.empty_cache()
    
    # Load model
    print("\nLoading model...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    feature_dim = ckpt.get('feature_dim', 256) if 'model_state_dict' in ckpt else ckpt['classifier.weight'].shape[1]
    model = LinearProbe(feature_dim, 2)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    if 'history' in ckpt:
        print(f"Training IoU: {ckpt['history']['val_ious'][-1]:.4f}")
    
    # Test
    test_dataset = HDF5Dataset(features_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    results = test_model(model, test_loader, device)
    
    # Save
    with open(os.path.join(args.save_dir, 'results.txt'), 'w') as f:
        f.write(f"Test on {args.num_patients} BraTS 2021 patients\n")
        f.write("="*60 + "\n")

        for k, v in results.items():
            if isinstance(v, (float, int)):
                f.write(f"{k}: {v:.4f}\n")
            elif k == 'confusion_matrix':
                f.write("confusion_matrix:\n")
                f.write(f"{v}\n")

    plot_metrics(results, args.save_dir)
    plot_predictions(features_path, model, args.save_dir, device, num_samples=5)
    create_plots(results, args.save_dir, args.num_patients)
    if 'history' in ckpt:
        print(f"\nComparison:")
        print(f"  Training: {ckpt['history']['val_ious'][-1]:.4f}")
        print(f"  Test:     {results['iou']:.4f}")
        print(f"  Gap:      {results['iou'] - ckpt['history']['val_ious'][-1]:+.4f}")
    
    print(f"\n✓ Complete! Results in {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bpe_path', required=True)
    parser.add_argument('--num_patients', type=int, default=125)
    parser.add_argument('--save_dir', default='./test_125_fixed')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
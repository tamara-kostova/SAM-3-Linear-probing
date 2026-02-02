#!/usr/bin/env python3
"""
Test Linear Probe on BraTS 2021

Simple script to test trained BraTS 2020 model on BraTS 2021 data.

Usage:
    python test_tumor_script.py \
        --test_data /path/to/BraTS2021_Training_Data \
        --checkpoint ./checkpoints/final_probe.pth \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --save_dir ./test_brats2021
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, jaccard_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from dataset import MedicalDataset


class LinearProbe(nn.Module):
    """Simple linear classifier for segmentation"""
    def __init__(self, feature_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    
    def forward(self, features):
        return self.classifier(features)


class SAM3FeatureExtractor:
    """Extract features from frozen SAM3"""
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
            batch_size = images.shape[0]
            captions = ["brain tumor"] * batch_size
        
        backbone_out = self.model.backbone(images, captions)
        
        if isinstance(backbone_out, dict):
            for key in ['vision_features', 'image_features', 'features']:
                if key in backbone_out:
                    return backbone_out[key]
            for v in backbone_out.values():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    return v
        
        return backbone_out


def precompute_test_features(test_dataset, feature_extractor, save_path,
                             batch_size=16, device='cuda'):
    """Pre-compute features for test set"""
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    print(f"\n{'='*80}")
    print("PRE-COMPUTING TEST FEATURES")
    print('='*80)
    print(f"Test samples: {len(test_dataset)}")
    print(f"Save path: {save_path}")
    
    # Get dimensions
    first_images, first_masks = next(iter(dataloader))
    first_images = first_images.to(device)
    
    with torch.no_grad():
        test_features = feature_extractor.extract_features(first_images)
    
    feature_shape = test_features.shape[1:]
    mask_shape = first_masks.shape[1:]
    total_samples = len(test_dataset)
    
    print(f"  Feature shape: {feature_shape}")
    print(f"  Total samples: {total_samples}\n")
    
    # Create HDF5 file
    with h5py.File(save_path, 'w') as h5f:
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
        
        h5f.attrs['feature_dim'] = feature_shape[0]
        h5f.attrs['total_samples'] = total_samples
        
        current_idx = 0
        
        for images, masks in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            with torch.no_grad():
                features = feature_extractor.extract_features(images)
                
                features_dset[current_idx:current_idx + batch_size_actual] = features.cpu().numpy()
                masks_dset[current_idx:current_idx + batch_size_actual] = masks.numpy()
                
                current_idx += batch_size_actual
                
                del features, images
                if current_idx % 160 == 0:
                    torch.cuda.empty_cache()
        
        print(f"\n✓ Saved {current_idx} test features")
    
    print(f"✓ File size: {os.path.getsize(save_path) / 1e6:.1f} MB\n")


class HDF5FeaturesDataset(torch.utils.data.Dataset):
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
        
        features = torch.from_numpy(self.h5f['features'][idx]).float()
        masks = torch.from_numpy(self.h5f['masks'][idx]).long()
        
        return features, masks


def test_model(model, test_loader, device='cuda', save_dir='./'):
    """Run comprehensive testing"""
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE ON TEST SET")
    print('='*80)
    
    with torch.no_grad():
        for features, masks in tqdm(test_loader, desc="Testing"):
            features = features.to(device)
            masks = masks.to(device)
            
            logits = model(features)
            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())
            all_probs.append(probs[:, 1].cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("TEST RESULTS ON BRATS 2021")
    print('='*80)
    
    accuracy = accuracy_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds, average='binary', zero_division=0)
    dice = 2 * iou / (1 + iou)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    
    # Specificity
    tn = np.sum((all_targets == 0) & (all_preds == 0))
    fp = np.sum((all_targets == 0) & (all_preds == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPixel-wise Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  IoU:         {iou:.4f}")
    print(f"  Dice:        {dice:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f"\nClass Distribution:")
    total_pixels = len(all_targets)
    tumor_pixels = np.sum(all_targets)
    bg_pixels = total_pixels - tumor_pixels
    print(f"  Background: {bg_pixels:,} ({bg_pixels/total_pixels*100:.2f}%)")
    print(f"  Tumor:      {tumor_pixels:,} ({tumor_pixels/total_pixels*100:.2f}%)")
    
    print('='*80)
    
    # Save results
    results_file = os.path.join(save_dir, 'brats2021_test_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS ON BRATS 2021\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Samples: {total_pixels}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Accuracy:    {accuracy:.4f}\n")
        f.write(f"  IoU:         {iou:.4f}\n")
        f.write(f"  Dice:        {dice:.4f}\n")
        f.write(f"  Precision:   {precision:.4f}\n")
        f.write(f"  Recall:      {recall:.4f}\n")
        f.write(f"  F1-Score:    {f1:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(all_targets, all_preds, all_probs, save_dir)
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }


def create_visualizations(targets, preds, probs, save_dir):
    """Create result visualizations"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Metrics bar chart
    metrics = {
        'Accuracy': accuracy_score(targets, preds),
        'IoU': jaccard_score(targets, preds, average='binary', zero_division=0),
        'Dice': 2 * jaccard_score(targets, preds, average='binary', zero_division=0) / 
                (1 + jaccard_score(targets, preds, average='binary', zero_division=0)),
        'Precision': precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)[0],
        'Recall': precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)[1],
        'F1': precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)[2]
    }
    
    axes[0].barh(list(metrics.keys()), list(metrics.values()), color='steelblue')
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('Score')
    axes[0].set_title('Test Metrics on BraTS 2021')
    axes[0].axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='0.7 threshold')
    axes[0].legend()
    
    for i, (metric, value) in enumerate(metrics.items()):
        axes[0].text(value + 0.02, i, f'{value:.3f}', va='center')
    
    # 2. Prediction distribution
    axes[1].hist(probs[targets == 0], bins=50, alpha=0.5, label='Background', color='blue')
    axes[1].hist(probs[targets == 1], bins=50, alpha=0.5, label='Tumor', color='red')
    axes[1].set_xlabel('Predicted Probability (Tumor Class)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Distribution')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'brats2021_test_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualizations saved to {save_path}")
    plt.close()


def main(args):
    """Main testing pipeline"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("TESTING ON BRATS 2021")
    print('='*80)
    print(f"Test data:   {args.test_data}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Device:      {device}")
    print('='*80)
    
    # Check how many patients
    patient_dirs = glob.glob(os.path.join(args.test_data, "BraTS*"))
    print(f"\n✓ Found {len(patient_dirs)} patients in test data")
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = MedicalDataset(
        data_root=args.test_data,
        dataset_type='brats',
        modality='t1ce',
        cache_dir=os.path.join(args.save_dir, "test_cache"),
        force_rebuild=args.rebuild_cache
    )
    
    print(f"✓ Loaded {len(test_dataset)} test slices")
    
    # Extract features
    test_features_path = os.path.join(args.save_dir, "brats2021_features.h5")
    
    if not os.path.exists(test_features_path) or args.rebuild_cache:
        from sam3.sam3 import build_sam3_image_model
        
        print("\nLoading SAM3 model...")
        sam3_model = build_sam3_image_model(bpe_path=args.bpe_path)
        feature_extractor = SAM3FeatureExtractor(sam3_model, device=device)
        
        precompute_test_features(
            test_dataset, feature_extractor,
            test_features_path, batch_size=16, device=device
        )
        
        del sam3_model, feature_extractor
        torch.cuda.empty_cache()
    else:
        print(f"\n✓ Using existing features: {test_features_path}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        feature_dim = checkpoint.get('feature_dim', 256)
        model = LinearProbe(feature_dim=feature_dim, num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training info if available
        if 'history' in checkpoint:
            train_iou = checkpoint['history']['val_ious'][-1]
            print(f"  Model was trained to IoU: {train_iou:.4f} on BraTS 2020")
    else:
        feature_dim = checkpoint['classifier.weight'].shape[1]
        model = LinearProbe(feature_dim=feature_dim, num_classes=2)
        model.load_state_dict(checkpoint)
    
    print(f"✓ Model loaded (feature_dim: {feature_dim})")
    
    # Load features
    test_features_dataset = HDF5FeaturesDataset(test_features_path)
    test_loader = DataLoader(
        test_features_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run testing
    results = test_model(model, test_loader, device, args.save_dir)
    
    # Compare with training
    if 'history' in checkpoint:
        train_iou = checkpoint['history']['val_ious'][-1]
        test_iou = results['iou']
        
        print(f"\n{'='*80}")
        print("GENERALIZATION ANALYSIS")
        print('='*80)
        print(f"BraTS 2020 (Training Val) IoU:  {train_iou:.4f}")
        print(f"BraTS 2021 (Test) IoU:          {test_iou:.4f}")
        print(f"Generalization Gap:             {test_iou - train_iou:+.4f} ({(test_iou - train_iou)/train_iou*100:+.2f}%)")
        
        if abs(test_iou - train_iou) < 0.02:
            print("\n✓ Excellent generalization! (< 2% gap)")
        elif abs(test_iou - train_iou) < 0.04:
            print("\n✓ Good generalization (2-4% gap)")
        else:
            print("\n⚠ Notable performance gap (> 4%)")
        
        print('='*80)
    
    print(f"\n✓ Testing complete!")
    print(f"\nResults saved to: {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test linear probe on BraTS 2021'
    )
    
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to BraTS2021_Training_Data folder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--bpe_path', type=str, required=True,
                        help='Path to BPE vocabulary')
    parser.add_argument('--save_dir', type=str, default='./test_brats2021',
                        help='Directory to save results')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_cache', action='store_true',
                        help='Force rebuild cache and features')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
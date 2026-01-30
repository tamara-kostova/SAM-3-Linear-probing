#!/usr/bin/env python3
"""
Universal Medical Segmentation Dataset
Supports: BraTS (tumors), MS (lesions), Stroke (lesions)

This replaces PreprocessedBraTSDataset with a more flexible version.
"""

import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm


class MedicalDataset(Dataset):
    """
    Universal dataset for medical image segmentation.
    Supports: BraTS, MS, Stroke datasets with automatic file detection.
    """
    def __init__(self, data_root, dataset_type='brats', modality='flair', 
                 cache_dir='./cache', img_size=1008, force_rebuild=False,
                 slice_range=None):
        """
        Args:
            data_root: Path to dataset root directory
            dataset_type: 'brats', 'ms', or 'stroke'
            modality: Image modality to use
            cache_dir: Directory to cache preprocessed slices
            img_size: Target image size (1008 for SAM3)
            force_rebuild: Force rebuild cache even if exists
            slice_range: (start, end) slice indices (None = use all slices with lesions)
        """
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_index_file = os.path.join(cache_dir, f'{dataset_type}_slice_paths.txt')
        
        if not force_rebuild and os.path.exists(cache_index_file):
            print(f"Loading {dataset_type.upper()} dataset from cache...")
            with open(cache_index_file, 'r') as f:
                self.cache_paths = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(self.cache_paths)} cached slices")
        else:
            print(f"Building {dataset_type.upper()} dataset cache...")
            self.cache_paths = self._preprocess_and_cache(
                data_root, dataset_type, modality, img_size, slice_range
            )
            with open(cache_index_file, 'w') as f:
                f.write('\n'.join(self.cache_paths))
            print(f"✓ Cache built: {len(self.cache_paths)} slices")
    
    def _find_files(self, patient_dir, dataset_type, modality):
        """
        Automatically find image and mask files based on dataset type.
        Returns: (image_path, mask_path) or (None, None) if not found
        """
        patient_id = os.path.basename(patient_dir)
        
        # Define search patterns for each dataset type
        if dataset_type == 'brats':
            img_patterns = [
                f"{patient_id}_{modality}.nii",
                f"{patient_id}_{modality}.nii.gz",
                f"*{modality}*.nii*"
            ]
            mask_patterns = [
                f"{patient_id}_seg.nii",
                f"{patient_id}_seg.nii.gz",
                "*seg*.nii*"
            ]
        
        elif dataset_type == 'ms':
            # MS: typically FLAIR + lesion mask
            img_patterns = [
                f"{patient_id}_{modality}.nii.gz",
                f"{patient_id}_{modality}.nii",
                f"*{modality}*.nii*",
                "*FLAIR*.nii*",
                "*flair*.nii*"
            ]
            mask_patterns = [
                f"{patient_id}_mask.nii.gz",
                f"{patient_id}_lesion.nii.gz",
                "*mask*.nii*",
                "*lesion*.nii*",
                "*seg*.nii*",
                "*Consensus*.nii*"  # MSLesSeg specific
            ]
        
        elif dataset_type == 'stroke':
            # Stroke: DWI/ADC + lesion mask
            img_patterns = [
                f"{patient_id}_{modality}.nii.gz",
                f"{patient_id}_{modality}.nii",
                f"*{modality}*.nii*",
                "*DWI*.nii*",
                "*dwi*.nii*",
                "*ADC*.nii*"
            ]
            mask_patterns = [
                f"{patient_id}_mask.nii.gz",
                f"{patient_id}_lesion.nii.gz",
                "*mask*.nii*",
                "*lesion*.nii*",
                "*msk*.nii*"
            ]
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Search for image file
        img_path = None
        for pattern in img_patterns:
            full_pattern = os.path.join(patient_dir, pattern)
            matches = glob.glob(full_pattern)
            if matches:
                img_path = matches[0]
                break
        
        # Search for mask file
        mask_path = None
        for pattern in mask_patterns:
            full_pattern = os.path.join(patient_dir, pattern)
            matches = glob.glob(full_pattern)
            if matches:
                mask_path = matches[0]
                break
        
        return img_path, mask_path
    
    def _preprocess_and_cache(self, data_root, dataset_type, modality, img_size, slice_range):
        """Preprocess all slices and save to disk"""
        
        # Find all patient directories
        patient_dirs = sorted([d for d in glob.glob(os.path.join(data_root, "*")) 
                              if os.path.isdir(d)])
        
        if len(patient_dirs) == 0:
            raise ValueError(f"No patient directories found in {data_root}")
        
        print(f"Found {len(patient_dirs)} patient directories")
        
        cache_paths = []
        skipped_patients = []
        
        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            patient_id = os.path.basename(patient_dir)
            
            # Find image and mask files
            img_path, mask_path = self._find_files(patient_dir, dataset_type, modality)
            
            if img_path is None or mask_path is None:
                skipped_patients.append(patient_id)
                continue
            
            # Load volumes
            try:
                img_data = nib.load(img_path).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
            except Exception as e:
                print(f"\n  Warning: Failed to load {patient_id}: {e}")
                skipped_patients.append(patient_id)
                continue
            
            # Check shape compatibility
            if img_data.shape != mask_data.shape:
                print(f"\n  Warning: Shape mismatch for {patient_id}")
                print(f"    Image: {img_data.shape}, Mask: {mask_data.shape}")
                skipped_patients.append(patient_id)
                continue
            
            # Determine slice range
            num_slices = img_data.shape[2]
            if slice_range is None:
                # For MS/Stroke: use only slices with lesions
                if dataset_type in ['ms', 'stroke']:
                    valid_slices = []
                    for s in range(num_slices):
                        if mask_data[:, :, s].max() > 0:
                            valid_slices.append(s)
                    slice_indices = valid_slices
                else:
                    # For BraTS: use predefined range
                    slice_indices = range(50, min(130, num_slices))
            else:
                slice_indices = range(slice_range[0], min(slice_range[1], num_slices))
            
            # Process each slice
            for slice_idx in slice_indices:
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Skip empty slices for MS/Stroke
                if dataset_type in ['ms', 'stroke'] and mask_slice.max() == 0:
                    continue
                
                # Normalize image
                img_min, img_max = img_slice.min(), img_slice.max()
                if img_max > img_min:
                    img_slice = (img_slice - img_min) / (img_max - img_min)
                else:
                    # Skip if image is uniform (likely corrupted)
                    continue
                
                # Convert to RGB (SAM3 expects 3 channels)
                img_slice_rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
                img_tensor = torch.from_numpy(img_slice_rgb).float().permute(2, 0, 1)
                
                # Binary mask (lesion/tumor vs background)
                mask_binary = (mask_slice > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_binary).float()
                
                # Resize to SAM3 input size
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=(img_size, img_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0), size=(img_size, img_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
                
                # Save to cache
                cache_path = os.path.join(
                    self.cache_dir,
                    f"{dataset_type}_{patient_id}_{slice_idx:03d}.pt"
                )
                torch.save({'image': img_tensor, 'mask': mask_tensor}, cache_path)
                cache_paths.append(cache_path)
        
        # Report statistics
        if skipped_patients:
            print(f"\n⚠ Skipped {len(skipped_patients)} patients due to missing/invalid files")
            if len(skipped_patients) <= 10:
                for p in skipped_patients:
                    print(f"    - {p}")
        
        if len(cache_paths) == 0:
            raise ValueError(
                f"No valid slices found! Check your dataset structure.\n"
                f"Expected modality: {modality}\n"
                f"Dataset type: {dataset_type}"
            )
        
        return cache_paths
    
    def __len__(self):
        return len(self.cache_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.cache_paths[idx])
        return data['image'], data['mask']


# Example usage
if __name__ == '__main__':
    # Test BraTS
    brats_dataset = MedicalDataset(
        data_root='/path/to/BraTS2020',
        dataset_type='brats',
        modality='t1ce',
        cache_dir='./cache_brats'
    )
    print(f"BraTS: {len(brats_dataset)} slices")
    
    # Test MS
    ms_dataset = MedicalDataset(
        data_root='/path/to/MSLesSeg',
        dataset_type='ms',
        modality='flair',
        cache_dir='./cache_ms'
    )
    print(f"MS: {len(ms_dataset)} slices")
    
    # Test Stroke
    stroke_dataset = MedicalDataset(
        data_root='/path/to/ISLES2022',
        dataset_type='stroke',
        modality='dwi',
        cache_dir='./cache_stroke'
    )
    print(f"Stroke: {len(stroke_dataset)} slices")
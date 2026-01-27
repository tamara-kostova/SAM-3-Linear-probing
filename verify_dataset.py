#!/usr/bin/env python3
"""
BraTS 2020 Dataset Verification Script
======================================

Verifies the integrity and completeness of the BraTS 2020 dataset.

Usage:
    python verify_dataset.py /path/to/MICCAI_BraTS2020_TrainingData
"""

import os
import sys
import glob
import argparse

def verify_brats_dataset(data_root, verbose=True):
    """
    Verify BraTS dataset integrity
    
    Args:
        data_root: Path to MICCAI_BraTS2020_TrainingData directory
        verbose: Print detailed information
    
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    
    if not os.path.exists(data_root):
        print(f"❌ Error: Directory not found: {data_root}")
        return False
    
    patient_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS20_Training_*")))
    
    print("="*80)
    print("BraTS 2020 Dataset Verification")
    print("="*80)
    print(f"Dataset root: {data_root}")
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Check patient count
    expected_patients = 369
    if len(patient_dirs) != expected_patients:
        print(f"⚠ Warning: Expected {expected_patients} patients, found {len(patient_dirs)}")
    else:
        print(f"✓ Patient count correct: {len(patient_dirs)}")
    
    if len(patient_dirs) == 0:
        print("❌ Error: No patient directories found!")
        print("\nExpected structure:")
        print("  MICCAI_BraTS2020_TrainingData/")
        print("  ├── BraTS20_Training_001/")
        print("  ├── BraTS20_Training_002/")
        print("  └── ...")
        return False
    
    # Check first patient in detail
    print("\n" + "="*80)
    print("Sample Patient Verification")
    print("="*80)
    
    sample_dir = patient_dirs[0]
    patient_id = os.path.basename(sample_dir)
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']
    
    print(f"Checking: {patient_id}")
    
    sample_valid = True
    for mod in modalities:
        filepath = os.path.join(sample_dir, f"{patient_id}_{mod}.nii")
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1e6
            
            # Try to load with nibabel if available
            try:
                import nibabel as nib
                img = nib.load(filepath)
                shape = img.shape
                print(f"  ✓ {mod:6s}: {shape} ({size_mb:.1f} MB)")
            except ImportError:
                print(f"  ✓ {mod:6s}: ({size_mb:.1f} MB) [nibabel not available for shape check]")
            except Exception as e:
                print(f"  ⚠ {mod:6s}: File exists but could not be loaded: {e}")
                sample_valid = False
        else:
            print(f"  ❌ {mod:6s}: MISSING")
            sample_valid = False
    
    if not sample_valid:
        print("\n❌ Sample patient verification failed!")
        return False
    
    # Check all patients
    print("\n" + "="*80)
    print("Complete Dataset Verification")
    print("="*80)
    
    incomplete = []
    corrupted = []
    
    print("Checking all patients...")
    
    for i, patient_dir in enumerate(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        
        # Progress indicator
        if verbose and (i + 1) % 50 == 0:
            print(f"  Checked {i + 1}/{len(patient_dirs)} patients...")
        
        patient_incomplete = []
        for mod in modalities:
            filepath = os.path.join(patient_dir, f"{patient_id}_{mod}.nii")
            
            if not os.path.exists(filepath):
                patient_incomplete.append(mod)
            elif os.path.getsize(filepath) < 1000:  # Suspiciously small file
                corrupted.append((patient_id, mod, os.path.getsize(filepath)))
        
        if patient_incomplete:
            incomplete.append((patient_id, patient_incomplete))
    
    # Report results
    print(f"\nChecked {len(patient_dirs)} patients")
    
    if incomplete:
        print(f"\n❌ Found {len(incomplete)} patients with missing files:")
        for patient, missing_mods in incomplete[:10]:  # Show first 10
            print(f"  {patient}: missing {', '.join(missing_mods)}")
        if len(incomplete) > 10:
            print(f"  ... and {len(incomplete) - 10} more")
        dataset_valid = False
    else:
        print("✓ All patients have complete file sets")
        dataset_valid = True
    
    if corrupted:
        print(f"\n⚠ Found {len(corrupted)} suspiciously small files:")
        for patient, mod, size in corrupted[:10]:
            print(f"  {patient}_{mod}: {size} bytes")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    total_files = 0
    total_size = 0
    
    for patient_dir in patient_dirs:
        files = glob.glob(os.path.join(patient_dir, "*.nii"))
        total_files += len(files)
        total_size += sum(os.path.getsize(f) for f in files)
    
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Average per patient: {total_size / len(patient_dirs) / 1e6:.1f} MB")
    
    # Modality statistics
    print("\nExpected files per patient: 5 (flair, t1, t1ce, t2, seg)")
    print(f"Actual files per patient: {total_files / len(patient_dirs):.1f}")
    
    # Final verdict
    print("\n" + "="*80)
    if dataset_valid and not corrupted:
        print("✓ DATASET VERIFICATION PASSED")
        print("="*80)
        return True
    else:
        print("❌ DATASET VERIFICATION FAILED")
        print("="*80)
        if incomplete:
            print("\nAction required: Re-download or obtain missing files")
        if corrupted:
            print("\nAction required: Re-download corrupted files")
        return False


def check_prerequisites():
    """Check if required packages are available"""
    print("Checking prerequisites...")
    
    missing = []
    
    # Check nibabel
    try:
        import nibabel as nib
        print("  ✓ nibabel installed")
    except ImportError:
        print("  ⚠ nibabel not installed (optional, but recommended)")
        print("    Install with: pip install nibabel")
        missing.append("nibabel")
    
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify BraTS 2020 dataset integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_brats_dataset.py ~/brats_data/MICCAI_BraTS2020_TrainingData
  python verify_brats_dataset.py /data/BraTS2020/ --quiet
        """
    )
    
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to MICCAI_BraTS2020_TrainingData directory'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    print()
    check_prerequisites()
    print()
    
    # Verify dataset
    success = verify_brats_dataset(args.data_root, verbose=not args.quiet)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
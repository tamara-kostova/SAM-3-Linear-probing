#!/usr/bin/env python3
"""
Dataset loaders for MS Lesion (MSLesSeg) and Stroke (ISLES 2022) datasets.

Directory structures expected:
  MSLesSeg:
    <data_root>/
      Patient_*/  (or patient_*, sub-*, P*)
        *FLAIR*.nii[.gz]   (or T2_FLAIR, etc.)
        *T1*.nii[.gz]
        *lesion*.nii[.gz]  (or *mask*, *seg*)

  ISLES 2022 (strokedata.io layout):
    <data_root>/
      sub-*/
        ses-*/
          dwi/
            *dwi*.nii.gz
          anat/
            *FLAIR*.nii.gz  (optional)
          *lesion*.nii.gz   (or derivatives/sub-*/ses-*/*msk*.nii.gz)
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def normalize_slice(arr):
    """Robust min-max normalisation to [0, 1]."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def to_rgb_tensor(slice_2d, img_size):
    """Convert a 2-D float array to a [3, H, W] float32 tensor at img_size."""
    t = torch.from_numpy(slice_2d).float()
    rgb = t.unsqueeze(0).repeat(3, 1, 1)          # [3, H, W]
    rgb = F.interpolate(
        rgb.unsqueeze(0), size=(img_size, img_size),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    return rgb


def to_mask_tensor(mask_2d, img_size):
    """Convert a 2-D mask array to a [H, W] long tensor at img_size."""
    t = torch.from_numpy((mask_2d > 0).astype(np.float32))
    t = F.interpolate(
        t.unsqueeze(0).unsqueeze(0), size=(img_size, img_size),
        mode='nearest'
    ).squeeze(0).squeeze(0).long()
    return t


def find_file(directory, patterns):
    """Return the first file in `directory` matching any glob pattern."""
    if not os.path.isdir(directory):
        return None

    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        for m in matches:
            # Some archives unpack as "<name>.nii/<inner>.nii".
            # Prefer a real file; if a directory matches, look inside it.
            if os.path.isfile(m):
                return m
            if os.path.isdir(m):
                nested = sorted(
                    glob.glob(os.path.join(m, '*.nii')) +
                    glob.glob(os.path.join(m, '*.nii.gz'))
                )
                if nested:
                    return nested[0]
    return None


# ─────────────────────────────────────────────────────────────────
# MS LESION DATASET  (MSLesSeg / MSSEG / similar BIDS-like)
# ─────────────────────────────────────────────────────────────────

class MSLesSegDataset(Dataset):
    """
    Linear-probe dataset for MS lesion segmentation.

    Key differences from BraTS:
      • Much smaller lesions  →  high class imbalance (use weighted loss)
      • Modality is usually FLAIR; T1 available for registration reference
      • Some slices have NO lesion – we keep a configurable fraction of
        empty slices so the probe sees realistic class ratios
      • Binary labels: 0 = background, 1 = lesion

    Args:
        data_root      : root folder containing patient sub-directories
        cache_dir      : where to store pre-processed .pt tensors
        modality       : 'flair' | 't1' (default: 'flair')
        img_size       : spatial size fed to SAM3 (default: 1008)
        empty_ratio    : fraction of empty (no-lesion) slices to keep
                         e.g. 0.3 means keep 30 % of empty slices
        force_rebuild  : rebuild cache even if it already exists
    """

    # Patterns used to find image / mask files inside a patient folder
    _IMG_PATTERNS = {
        'flair': ['*FLAIR*.nii.gz', '*flair*.nii.gz', '*T2_FLAIR*.nii.gz',
                  '*FLAIR*.nii',    '*flair*.nii'],
        't1':    ['*T1*.nii.gz',    '*t1*.nii.gz',
                  '*T1*.nii',       '*t1*.nii'],
    }
    _MASK_PATTERNS = ['*lesion*.nii.gz', '*mask*.nii.gz', '*seg*.nii.gz',
                      '*Lesion*.nii.gz', '*lesion*.nii',  '*mask*.nii',
                      '*MASK*.nii.gz',   '*MASK*.nii',    '*SEG*.nii.gz']

    def __init__(self, data_root, cache_dir,
                 modality='flair', img_size=1008,
                 empty_ratio=0.3, force_rebuild=False):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        index_file = os.path.join(cache_dir, 'slice_paths.txt')

        if not force_rebuild and os.path.exists(index_file):
            with open(index_file) as f:
                self.cache_paths = [l.strip() for l in f if l.strip()]
            print(f"✓ MS cache loaded: {len(self.cache_paths)} slices")
        else:
            print("Building MS cache (first run only) …")
            self.cache_paths = self._build_cache(
                data_root, modality, img_size, empty_ratio
            )
            with open(index_file, 'w') as f:
                f.write('\n'.join(self.cache_paths))
            print(f"✓ MS cache built: {len(self.cache_paths)} slices")

    # ── internal ────────────────────────────────────────────────

    def _build_cache(self, data_root, modality, img_size, empty_ratio):
        patient_dirs = sorted(
            glob.glob(os.path.join(data_root, 'Patient_*')) +
            glob.glob(os.path.join(data_root, 'patient_*')) +
            glob.glob(os.path.join(data_root, 'sub-*')) +      # BIDS-style
            glob.glob(os.path.join(data_root, 'P[0-9]*')) +    # MSLesSeg style
            glob.glob(os.path.join(data_root, 'p[0-9]*'))
        )
        if not patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {data_root}.\n"
                "Expected folders named Patient_*, patient_*, sub-*, or P*."
            )

        cache_paths = []
        patterns = self._IMG_PATTERNS.get(modality, self._IMG_PATTERNS['flair'])

        for pdir in tqdm(patient_dirs, desc='Processing MS patients'):
            pid = os.path.basename(pdir)
            rng = np.random.default_rng(seed=42)
            # Support both flat patient folders and per-visit subfolders (e.g. T1/T2/T3).
            scan_dirs = [pdir] + [
                d for d in sorted(glob.glob(os.path.join(pdir, '*')))
                if os.path.isdir(d)
            ]
            found_any = False

            for scan_dir in scan_dirs:
                img_path  = find_file(scan_dir, patterns)
                mask_path = find_file(scan_dir, self._MASK_PATTERNS)

                if img_path is None or mask_path is None:
                    continue

                found_any = True
                sid = os.path.basename(scan_dir)
                sample_prefix = pid if scan_dir == pdir else f"{pid}_{sid}"

                img_vol  = nib.load(img_path).get_fdata()
                mask_vol = nib.load(mask_path).get_fdata()
                n_slices = img_vol.shape[2]

                for s in range(n_slices):
                    img_sl  = normalize_slice(img_vol[:, :, s])
                    mask_sl = mask_vol[:, :, s]
                    has_lesion = mask_sl.any()

                    # Subsample empty slices to avoid overwhelming background
                    if not has_lesion:
                        if rng.random() > empty_ratio:
                            continue

                    img_t  = to_rgb_tensor(img_sl, img_size)
                    mask_t = to_mask_tensor(mask_sl, img_size)

                    out = os.path.join(self.cache_dir, f"{sample_prefix}_{s:03d}.pt")
                    torch.save({'image': img_t, 'mask': mask_t,
                                'has_lesion': has_lesion}, out)
                    cache_paths.append(out)

            if not found_any:
                print(f"  ⚠ Skipping {pid}: missing image or mask")

        return cache_paths

    # ── Dataset API ─────────────────────────────────────────────

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        d = torch.load(self.cache_paths[idx], weights_only=False)
        return d['image'], d['mask']

    # ── Utility ─────────────────────────────────────────────────

    def compute_class_weights(self):
        """
        Scan cache to compute pixel-level class weights.
        Useful for weighted CrossEntropyLoss to handle small-lesion imbalance.
        Returns a 2-element tensor [w_background, w_lesion].
        """
        counts = np.zeros(2, dtype=np.int64)
        for p in tqdm(self.cache_paths, desc='Computing class weights'):
            d = torch.load(p, weights_only=False)
            mask = d['mask'].numpy()
            counts[0] += (mask == 0).sum()
            counts[1] += (mask == 1).sum()
        total = counts.sum()
        # Inverse-frequency weighting
        weights = total / (2.0 * counts)
        print(f"  Background pixels: {counts[0]:,}  weight={weights[0]:.2f}")
        print(f"  Lesion pixels:     {counts[1]:,}  weight={weights[1]:.2f}")
        return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# STROKE DATASET  (ISLES 2022 / ISLES 2024)
# ─────────────────────────────────────────────────────────────────

class ISLESStrokeDataset(Dataset):
    """
    Linear-probe dataset for ischemic stroke lesion segmentation.

    Key differences from BraTS / MS:
      • ISLES 2022 uses DWI as primary modality (sometimes ADC or FLAIR)
      • Lesions can be very small (lacunar) or very large (MCA territory)
      • BIDS-style directory layout with derivatives for masks
      • Binary labels: 0 = background, 1 = infarct

    Supported layouts
    -----------------
    Layout A – ISLES 2022 official (strokedata.io):
        <data_root>/
          rawdata/sub-*/ses-*/dwi/*dwi*.nii.gz
          derivatives/sub-*/ses-*/sub-*_ses-*_msk.nii.gz

    Layout B – flat / custom:
        <data_root>/
          sub-*/
            *dwi*.nii.gz  (or *DWI*, *adc*, etc.)
            *lesion*.nii.gz

    Args:
        data_root   : see above
        cache_dir   : pre-processed tensor cache
        modality    : 'dwi' | 'adc' | 'flair'  (default: 'dwi')
        img_size    : spatial size (default: 1008)
        empty_ratio : fraction of empty slices to retain
        force_rebuild
    """

    _IMG_PATTERNS = {
        'dwi':   ['*dwi*.nii.gz',  '*DWI*.nii.gz',  '*dwi*.nii',  '*DWI*.nii'],
        'adc':   ['*adc*.nii.gz',  '*ADC*.nii.gz',  '*adc*.nii',  '*ADC*.nii'],
        'flair': ['*flair*.nii.gz','*FLAIR*.nii.gz', '*flair*.nii','*FLAIR*.nii'],
    }
    _MASK_PATTERNS = ['*msk*.nii.gz', '*lesion*.nii.gz', '*mask*.nii.gz',
                      '*seg*.nii.gz',  '*Lesion*.nii.gz',
                      '*msk*.nii',     '*lesion*.nii']

    def __init__(self, data_root, cache_dir,
                 modality='dwi', img_size=1008,
                 empty_ratio=0.3, force_rebuild=False):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        index_file = os.path.join(cache_dir, 'slice_paths.txt')

        if not force_rebuild and os.path.exists(index_file):
            with open(index_file) as f:
                self.cache_paths = [l.strip() for l in f if l.strip()]
            print(f"✓ Stroke cache loaded: {len(self.cache_paths)} slices")
        else:
            print("Building Stroke cache (first run only) …")
            self.cache_paths = self._build_cache(
                data_root, modality, img_size, empty_ratio
            )
            with open(index_file, 'w') as f:
                f.write('\n'.join(self.cache_paths))
            print(f"✓ Stroke cache built: {len(self.cache_paths)} slices")

    # ── internal ────────────────────────────────────────────────

    def _collect_pairs(self, data_root, modality):
        """
        Returns list of (img_path, mask_path, subject_id) tuples.
        Handles both ISLES 2022 official BIDS and flat layouts.
        """
        patterns = self._IMG_PATTERNS.get(modality, self._IMG_PATTERNS['dwi'])
        pairs = []

        # --- Layout A: official ISLES 2022 BIDS with rawdata/derivatives ---
        rawdata_root = os.path.join(data_root, 'rawdata')
        deriv_root   = os.path.join(data_root, 'derivatives')

        if os.path.isdir(rawdata_root):
            for sub in sorted(os.listdir(rawdata_root)):
                sub_dir = os.path.join(rawdata_root, sub)
                if not os.path.isdir(sub_dir):
                    continue
                for ses in sorted(os.listdir(sub_dir)):
                    ses_dir = os.path.join(sub_dir, ses)
                    dwi_dir = os.path.join(ses_dir, 'dwi')
                    anat_dir = os.path.join(ses_dir, 'anat')

                    # Find image (try dwi folder first, then anat)
                    img_path = (find_file(dwi_dir,  patterns) or
                                find_file(anat_dir, patterns) or
                                find_file(ses_dir,  patterns))

                    # Find mask in derivatives
                    deriv_ses = os.path.join(deriv_root, sub, ses)
                    mask_path = find_file(deriv_ses, self._MASK_PATTERNS)

                    if img_path and mask_path:
                        pairs.append((img_path, mask_path, f"{sub}_{ses}"))

        # --- Layout A2: BIDS-like root/sub-*/ses-* + derivatives ---
        # Common local layout: data_root/sub-*/ses-*/{dwi,anat}
        if not pairs:
            for sub_dir in sorted(glob.glob(os.path.join(data_root, 'sub-*'))):
                sub = os.path.basename(sub_dir)
                for ses_dir in sorted(glob.glob(os.path.join(sub_dir, 'ses-*'))):
                    ses = os.path.basename(ses_dir)
                    dwi_dir = os.path.join(ses_dir, 'dwi')
                    anat_dir = os.path.join(ses_dir, 'anat')

                    img_path = (find_file(dwi_dir,  patterns) or
                                find_file(anat_dir, patterns) or
                                find_file(ses_dir,  patterns))

                    deriv_ses = os.path.join(deriv_root, sub, ses)
                    mask_path = find_file(deriv_ses, self._MASK_PATTERNS)

                    if img_path and mask_path:
                        pairs.append((img_path, mask_path, f"{sub}_{ses}"))

        # --- Layout B: flat sub-* folders ---
        if not pairs:
            for sub_dir in sorted(glob.glob(os.path.join(data_root, 'sub-*'))):
                sub = os.path.basename(sub_dir)
                img_path  = find_file(sub_dir, patterns)
                mask_path = find_file(sub_dir, self._MASK_PATTERNS)
                if img_path and mask_path:
                    pairs.append((img_path, mask_path, sub))

        if not pairs:
            raise FileNotFoundError(
                f"No image-mask pairs found in {data_root}.\n"
                "Check that the folder matches ISLES 2022 BIDS or flat layout."
            )

        return pairs

    def _build_cache(self, data_root, modality, img_size, empty_ratio):
        pairs = self._collect_pairs(data_root, modality)
        cache_paths = []
        rng = np.random.default_rng(seed=42)

        for img_path, mask_path, sid in tqdm(pairs, desc='Processing stroke subjects'):
            img_vol  = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()

            # Handle 4-D DWI volumes (take b=1000 volume, typically index 1)
            if img_vol.ndim == 4:
                img_vol = img_vol[..., min(1, img_vol.shape[-1] - 1)]

            n_slices = img_vol.shape[2]

            for s in range(n_slices):
                img_sl  = normalize_slice(img_vol[:, :, s])
                mask_sl = mask_vol[:, :, s]
                has_lesion = mask_sl.any()

                if not has_lesion and rng.random() > empty_ratio:
                    continue

                img_t  = to_rgb_tensor(img_sl, img_size)
                mask_t = to_mask_tensor(mask_sl, img_size)

                out = os.path.join(self.cache_dir, f"{sid}_{s:03d}.pt")
                torch.save({'image': img_t, 'mask': mask_t,
                            'has_lesion': has_lesion}, out)
                cache_paths.append(out)

        return cache_paths

    # ── Dataset API ─────────────────────────────────────────────

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        d = torch.load(self.cache_paths[idx], weights_only=False)
        return d['image'], d['mask']
    
    def compute_class_weights(self):
        """
        Scan cache to compute pixel-level class weights.
        Useful for weighted CrossEntropyLoss to handle infarct/background imbalance.
        Returns a 2-element tensor [w_background, w_infarct].
        """
        counts = np.zeros(2, dtype=np.int64)
        for p in tqdm(self.cache_paths, desc='Computing class weights'):
            d = torch.load(p, weights_only=False)
            mask = d['mask'].numpy()
            counts[0] += (mask == 0).sum()
            counts[1] += (mask == 1).sum()

        total = counts.sum()
        weights = total / (2.0 * np.maximum(counts, 1))

        print(f"  Background pixels: {counts[0]:,}  weight={weights[0]:.2f}")
        print(f"  Infarct pixels:    {counts[1]:,}  weight={weights[1]:.2f}")
        return torch.tensor(weights, dtype=torch.float32)

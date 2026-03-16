import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, shapiro

# Load results
probe = pd.read_csv('test_125_fixed/per_case_dice.csv')
zeroshot = pd.read_csv('results/zero shot/zeroshot_brats21_results_checkpoint_per_slice.csv')

# The 7 failed cases had no valid mask — exclude them from both sides
failed = ['BraTS2021_00008', 'BraTS2021_00012', 'BraTS2021_00014',
          'BraTS2021_00024', 'BraTS2021_00085', 'BraTS2021_00485', 'BraTS2021_00496']

probe_matched    = probe[~probe['case'].isin(failed)].sort_values('case')
zeroshot_matched = zeroshot[~zeroshot['case'].isin(failed)].sort_values('case')

# Align on case ID to guarantee pairing
merged = probe_matched.merge(zeroshot_matched, on='case', suffixes=('_probe', '_zeroshot'))

probe_dice    = merged['dice_probe'].values
zeroshot_dice = merged['dice_zeroshot'].values

# 1. Wilcoxon signed-rank test (one-sided: probe > zero-shot)
stat, p = wilcoxon(probe_dice, zeroshot_dice, alternative='greater')
print(f"Wilcoxon: W={stat:.1f}, p={p:.4e}")

# 2. Summary stats
print(f"\nProbe    mean={probe_dice.mean():.3f}  median={np.median(probe_dice):.3f}  std={probe_dice.std():.3f}")
print(f"ZeroShot mean={zeroshot_dice.mean():.3f}  median={np.median(zeroshot_dice):.3f}  std={zeroshot_dice.std():.3f}")
print(f"n matched cases: {len(merged)}")

# 3. Optional: check if normal (to justify Wilcoxon vs t-test)
_, p_norm = shapiro(probe_dice - zeroshot_dice)
print(f"\nShapiro-Wilk on differences: p={p_norm:.4f} ({'normal' if p_norm > 0.05 else 'not normal — Wilcoxon correct'})")
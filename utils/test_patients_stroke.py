import torch
import os
from pathlib import Path

cache_dir = "results/cache_stroke"
all_paths = sorted(Path(cache_dir).glob("*.pt"))

# Reproduce the exact same split
full_n  = len(all_paths)
n_test  = int(full_n * 0.15)   # --test_ratio default
n_val   = int(full_n * 0.15)
n_train = full_n - n_val - n_test

gen = torch.Generator().manual_seed(42)
indices = torch.randperm(full_n, generator=gen).tolist()

test_indices = indices[n_train + n_val:]   # last chunk = test

# Map indices to subject IDs
test_subjects = set()
for i in test_indices:
    stem = all_paths[i].stem          # e.g. sub-stroke0001_ses-0001_042
    subject = "_".join(stem.split("_")[:2])  # sub-stroke0001_ses-0001
    test_subjects.add(subject)

print(sorted(test_subjects))
with open("stroke_test_subjects.txt", "w") as f:
    f.write("\n".join(sorted(test_subjects)))
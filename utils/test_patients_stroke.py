import os
import numpy as np
from pathlib import Path

data_root = Path("ISLES-2022")  # adjust to your path

# discover subjects
rawdata = data_root / "rawdata"
if rawdata.is_dir():
    all_subjects = sorted(d for d in os.listdir(rawdata) if d.startswith("sub-"))
else:
    all_subjects = sorted(
        d for d in os.listdir(data_root)
        if (data_root / d).is_dir() and d.startswith("sub-")
    )

print(f"Total subjects: {len(all_subjects)}")

# pick 25 test subjects with seed 42
rng = np.random.default_rng(42)
test_subjects = sorted(rng.choice(all_subjects, size=25, replace=False).tolist())

print(f"Selected {len(test_subjects)} test subjects:")
for s in test_subjects:
    print(f"  {s}")

# save to file
with open("stroke_test_subjects.txt", "w") as f:
    f.write("\n".join(test_subjects))
print("Saved to stroke_test_subjects.txt")
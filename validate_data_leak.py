import csv
import random
from pathlib import Path

BRATS21_MAPPING = Path('mappings/BraTS2021_MappingToTCIA.csv')
BRATS20_MAPPING = Path('mappings/name_mapping.csv')
OUTPUT_FILE = Path('results/patient_overlap/brats2021_non_overlap_125.csv')
OUTPUT_ALL_FILE = Path('results/patient_overlap/brats2021_non_overlap_all.csv')

# Collections with confirmed zero overlap with BraTS2020:
#   - UCSF-PDGM, CPTAC-GBM, IvyGAP, ACRIN: entirely new datasets
#   - Collection1/3/4/5/7/9: PatientID = "new-not-previously-in-TCIA"
SAFE_COLLECTIONS = {
    'UCSF-PDGM', 'UCSF-PDGM_Additional',
    'CPTAC-GBM',
    'IvyGAP',
    'ACRIN-FMISO-Brain (ACRIN 6684)',
    'Collection1', 'Collection3', 'Collection 4', 'Collection 5',
    'Collection 7', 'Collection 9',
}

# Same institution as BraTS2020 CBICA subjects — cannot verify without
# a direct patient-level mapping, so excluded from safe pool.
RISKY_COLLECTIONS = {'UPENN-GBM', 'UPENN-GBM_Additional'}

# Step 1: Load BraTS2020 TCGA IDs
brats20_tcga_ids = set()
brats20_total = 0
with open(BRATS20_MAPPING) as f:
    for row in csv.DictReader(f):
        brats20_total += 1
        tcga = row['TCGA_TCIA_subject_ID'].strip()
        if tcga != 'NA':
            brats20_tcga_ids.add(tcga)

print(f"BraTS2020 total subjects:          {brats20_total}")
print(f"BraTS2020 subjects with TCGA IDs:  {len(brats20_tcga_ids)}")

# Step 2: Load BraTS2021 Training subjects and classify each
counts = {'tcga_overlap': 0, 'tcga_new': 0, 'risky': 0, 'safe': 0, 'unknown': 0}
safe_pool = []

with open(BRATS21_MAPPING) as f:
    for row in csv.DictReader(f):
        if row['Segmentation (Task 1) Cohort'].strip() != 'Training':
            continue

        col = row['Data Collection (as on TCIA+additional)'].strip()
        pid = row['PatientID on TCIA Radiology Portal'].strip()
        brats21_id = row['BraTS2021 ID'].strip()
        entry = {'brats21_id': brats21_id, 'collection': col, 'patient_id': pid}

        if col in ('TCGA-GBM', 'TCGA-LGG'):
            if pid in brats20_tcga_ids:
                counts['tcga_overlap'] += 1          # confirmed in BraTS2020
            else:
                counts['tcga_new'] += 1              # TCGA but not in BraTS2020
                safe_pool.append(entry)
        elif col in RISKY_COLLECTIONS:
            counts['risky'] += 1
        elif col in SAFE_COLLECTIONS:
            counts['safe'] += 1
            safe_pool.append(entry)
        else:
            counts['unknown'] += 1
            print(f"  [WARN] unknown collection: {col!r} — excluded from safe pool")

print(f"\nBraTS2021 Training classification:")
print(f"  TCGA overlap with BraTS2020:          {counts['tcga_overlap']}")
print(f"  TCGA new (not in BraTS2020):          {counts['tcga_new']}")
print(f"  UPENN-GBM (risky, same inst. as CBICA): {counts['risky']}")
print(f"  Safe collections:                     {counts['safe']}")
if counts['unknown']:
    print(f"  Unknown/excluded:                     {counts['unknown']}")
print(f"  Safe pool total:                      {len(safe_pool)}")

assert len(safe_pool) >= 125, f"Safe pool too small: {len(safe_pool)}"

# Step 3: Sample 125
random.seed(42)
sampled = sorted(random.sample(safe_pool, 125), key=lambda x: x['brats21_id'])

print(f"\nSampled 125 from safe pool (seed=42)")
print("Preview:", [s['brats21_id'] for s in sampled[:5]])

# Step 4: Save
OUTPUT_FILE.parent.mkdir(exist_ok=True)
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['brats21_id', 'collection', 'patient_id'])
    writer.writeheader()
    writer.writerows(sampled)

print(f"Saved {len(sampled)} subjects → {OUTPUT_FILE}")

# Step 5: Save full non-overlap pool with sampled flag
sampled_ids = {s['brats21_id'] for s in sampled}
with open(OUTPUT_ALL_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['brats21_id', 'collection', 'patient_id', 'in_sample_125'])
    writer.writeheader()
    for entry in sorted(safe_pool, key=lambda x: x['brats21_id']):
        writer.writerow({**entry, 'in_sample_125': entry['brats21_id'] in sampled_ids})

print(f"Saved {len(safe_pool)} subjects → {OUTPUT_ALL_FILE}")

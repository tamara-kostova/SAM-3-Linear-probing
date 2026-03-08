#!/usr/bin/env python3
"""Intersect BraTS2021 patient directories with a CSV of patient IDs.

Default behavior:
1) Read rows from results/patient_overlap/brats2021_non_overlap_all.csv
2) Keep rows where `brats21_id` exists as a directory in
   brats2021/BraTS2021_Training_Data
3) Write filtered rows to
   results/patient_overlap/brats2021_non_overlap_all_intersection.csv
"""

import argparse
import csv
import os
from typing import List, Tuple


def load_existing_patient_dirs(data_root: str) -> set:
    return {
        name
        for name in os.listdir(data_root)
        if name.startswith("BraTS2021_") and os.path.isdir(os.path.join(data_root, name))
    }


def intersect_rows(input_csv: str, available_ids: set) -> Tuple[List[dict], List[str], List[str]]:
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "brats21_id" not in (reader.fieldnames or []):
            raise ValueError(f"Input CSV must contain 'brats21_id': {input_csv}")

        kept_rows: List[dict] = []
        missing_ids: List[str] = []
        seen_ids = set()

        for row in reader:
            patient_id = (row.get("brats21_id") or "").strip()
            if not patient_id:
                continue
            if patient_id in seen_ids:
                continue
            seen_ids.add(patient_id)

            if patient_id in available_ids:
                kept_rows.append(row)
            else:
                missing_ids.append(patient_id)

    return kept_rows, missing_ids, sorted(seen_ids)


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="brats2021/BraTS2021_Training_Data",
        help="Path to BraTS2021 patient directories",
    )
    parser.add_argument(
        "--input_csv",
        default="results/patient_overlap/brats2021_non_overlap_all.csv",
        help="CSV containing a brats21_id column",
    )
    parser.add_argument(
        "--output_csv",
        default="results/patient_overlap/brats2021_non_overlap_all_intersection.csv",
        help="Where to write the intersected rows",
    )
    parser.add_argument(
        "--missing_txt",
        default="results/patient_overlap/brats2021_non_overlap_all_missing_in_dataset.txt",
        help="Where to write IDs found in CSV but missing in data_root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_ids = load_existing_patient_dirs(args.data_root)

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    kept_rows, missing_ids, csv_ids = intersect_rows(args.input_csv, available_ids)
    write_csv(args.output_csv, kept_rows, fieldnames)

    os.makedirs(os.path.dirname(args.missing_txt) or ".", exist_ok=True)
    with open(args.missing_txt, "w") as f:
        for patient_id in missing_ids:
            f.write(patient_id + "\n")

    print(f"Data root IDs: {len(available_ids)}")
    print(f"CSV unique IDs: {len(csv_ids)}")
    print(f"Intersection: {len(kept_rows)}")
    print(f"Missing from data root: {len(missing_ids)}")
    print(f"Wrote intersection CSV: {args.output_csv}")
    print(f"Wrote missing IDs: {args.missing_txt}")


if __name__ == "__main__":
    main()

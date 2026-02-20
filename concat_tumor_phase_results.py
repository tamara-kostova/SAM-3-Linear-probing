#!/usr/bin/env python3
"""
Build a single tumor-only manifest JSON from all phase summary JSONL files.

This script reads MedGemma result JSONLs, filters to tumor datasets only,
deduplicates by image_path, and writes a compact JSON list that can be used by:
    run_sam3_segmentation.py --input_manifest <output_json>
"""

import os
import json
import glob
import argparse
from pathlib import Path

TUMOR_DATASETS = {"images-17", "images-44c", "figshare", "br35h"}


def is_tumor_dataset(dataset_name):
    return str(dataset_name or "").strip().lower() in TUMOR_DATASETS


def norm_label(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "pituitary": "pituitary_tumor",
        "pituitary_tumour": "pituitary_tumor",
        "papiloma": "papilloma",
        "astrocytoma": "glioma",
        "glioblastoma": "glioma",
        "oligodendroglioma": "glioma",
        "ependymoma": "glioma",
        "ganglioglioma": "glioma",
        "no_tumor": "normal",
        "notumor": "normal",
    }
    return aliases.get(text, text)


def expand_jsonl_paths(inputs):
    expanded = []
    seen = set()
    for raw in inputs:
        if not raw:
            continue
        if os.path.isdir(raw):
            candidates = sorted(glob.glob(os.path.join(raw, "*summary*.jsonl")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(raw, "*.jsonl")))
        elif any(t in raw for t in ["*", "?", "["]):
            candidates = sorted(glob.glob(raw))
        else:
            candidates = [raw]
        for p in candidates:
            if p not in seen:
                expanded.append(p)
                seen.add(p)
    return expanded


def resolve_image_path(image_path):
    raw = str(image_path or "").strip()
    if not raw:
        return None

    candidates = [raw]
    norm = raw.replace("\\", "/")

    if norm.startswith("./"):
        candidates.append(norm[2:])
    candidates.append(norm)

    if "data/raw data/" in norm:
        candidates.append(norm.replace("data/raw data/", "data/"))
    if "raw data/" in norm:
        candidates.append(norm.replace("raw data/", ""))

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand):
            return cand
    return None


def build_manifest(jsonl_paths, split_filter=None):
    rows = []
    seen_images = set()

    for path in expand_jsonl_paths(jsonl_paths):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                processing = obj.get("processing", {}) or {}
                if split_filter and processing.get("split") != split_filter:
                    continue

                meta = obj.get("input", {}).get("metadata", {}) or {}
                dataset = meta.get("dataset")
                if not is_tumor_dataset(dataset):
                    continue

                image_path_raw = obj.get("input", {}).get("image_path")
                image_path = resolve_image_path(image_path_raw)
                if not image_path or image_path in seen_images:
                    continue

                gt_name = norm_label(meta.get("class"))
                gt_subclass = norm_label(meta.get("subclass"))

                rows.append({
                    "image_path": image_path,
                    "image_path_raw": image_path_raw,
                    "dataset": dataset,
                    "gt_label": gt_subclass if gt_subclass else gt_name or "tumor",
                    "class": gt_name,
                    "subclass": gt_subclass,
                    "source_file": path,
                    "split": processing.get("split"),
                    "phase": processing.get("phase"),
                })
                seen_images.add(image_path)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate tumor phase JSONL results into one manifest JSON")
    parser.add_argument("--input_jsonl", nargs="+", default=["results/*summary*.jsonl"],
                        help="JSONL files/globs/dirs (default: results/*summary*.jsonl)")
    parser.add_argument("--output_json", default="results/tumor_phase_manifest.json",
                        help="Output manifest JSON path")
    parser.add_argument("--medgemma_split", default=None,
                        help="Optional split filter, e.g. subset_1")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_manifest(args.input_jsonl, split_filter=args.medgemma_split)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[OK] Wrote {len(rows)} tumor records to {output_path}")


if __name__ == "__main__":
    main()

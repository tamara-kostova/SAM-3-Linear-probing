#!/usr/bin/env python3
"""
Phase 2: MedGemma Evaluation

Reads pre-segmented masked images from the Phase 1 manifest,
sends them to a running vLLM MedGemma endpoint, and computes metrics.

Start vLLM first:
    python -m vllm.entrypoints.openai.api_server \
        --model google/medgemma-1.5-4b-it \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --port 8000

Then run this script:
    python run_medgemma_eval.py \
        --manifest ./segmented/segmentation_manifest_<timestamp>.json \
        --output_dir ./results/tumor_evaluation \
        --endpoint http://localhost:8000/v1/chat/completions

Optionally compare against a standalone (no-SAM3) baseline from a previous
MedGemma JSONL run:
    python run_medgemma_eval.py \
        --manifest ./segmented/segmentation_manifest_<timestamp>.json \
        --standalone_jsonl ./results/standalone/*.jsonl \
        --output_dir ./results/tumor_evaluation
"""

import os
import json
import argparse
import glob
import base64
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

DEFAULT_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = "google/medgemma-1.5-4b-it"

TUMOR_DATASETS = {"images-17", "images-44c", "figshare", "br35h"}


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _norm_label(value):
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


def _extract_prediction_label(prediction):
    if not isinstance(prediction, dict):
        return None
    detailed = _norm_label(prediction.get("diagnosis_detailed"))
    if detailed:
        return detailed
    return _norm_label(prediction.get("diagnosis_name"))


def _parse_json_payload(content):
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for chunk in [c.strip() for c in text.split("```") if c.strip()]:
            candidate = chunk[4:].strip() if chunk.startswith("json") else chunk
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _is_tumor_dataset(dataset_name):
    return str(dataset_name or "").strip().lower() in TUMOR_DATASETS


def _expand_jsonl_paths(jsonl_paths):
    expanded, seen = [], set()
    for raw_path in (jsonl_paths or []):
        if not raw_path:
            continue
        if os.path.isdir(raw_path):
            candidates = sorted(glob.glob(os.path.join(raw_path, "*summary*.jsonl"))) or \
                         sorted(glob.glob(os.path.join(raw_path, "*.jsonl")))
        elif any(t in raw_path for t in ["*", "?", "["]):
            candidates = sorted(glob.glob(raw_path))
        else:
            candidates = [raw_path]
        for p in candidates:
            if p not in seen:
                expanded.append(p)
                seen.add(p)
    return expanded


# ============================================================================
# STANDALONE JSONL LOADER  (optional baseline)
# ============================================================================

def load_standalone_from_jsonl(jsonl_paths, split_filter=None, tumor_only=True):
    rows = []
    for path in _expand_jsonl_paths(jsonl_paths):
        if not os.path.exists(path):
            print(f"Warning: JSONL not found: {path}")
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
                if tumor_only and not _is_tumor_dataset(meta.get("dataset")):
                    continue

                output = obj.get("output", {}) or {}
                parsed = output.get("parsed_response", {}) or {}
                if output.get("status") != "success" or not isinstance(parsed, dict) or not parsed:
                    continue

                gt_name = _norm_label(meta.get("class"))
                gt_subclass = _norm_label(meta.get("subclass"))

                rows.append({
                    "image_path": obj.get("input", {}).get("image_path"),
                    "gt_label": gt_subclass if gt_subclass else gt_name,
                    "prediction": parsed,
                    "mode": "standalone_jsonl",
                    "source_file": path,
                    "split": processing.get("split"),
                })
    return rows


# ============================================================================
# MEDGEMMA CLIENT
# ============================================================================

class MedGemmaAnalyzer:
    def __init__(self, endpoint=DEFAULT_ENDPOINT, model=MODEL_NAME):
        self.endpoint = endpoint
        self.model = model

    def encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def analyze(self, image, system_prompt=SYSTEM_PROMPT, timeout=30):
        image_b64 = self.encode_image(image)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ],
                },
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = _parse_json_payload(content)
            if parsed is None:
                raise ValueError("Failed to parse model JSON output")
            return parsed
        except Exception as e:
            print(f"Warning MedGemma error: {e}")
            return None


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest from Phase 1
    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    if args.max_samples:
        manifest = manifest[:args.max_samples]

    print(f"\n{'='*80}")
    print(f"MEDGEMMA EVALUATION  —  {len(manifest)} masked images")
    print(f"Endpoint: {args.endpoint}")
    print("="*80 + "\n")

    analyzer = MedGemmaAnalyzer(endpoint=args.endpoint, model=args.model)
    results_sam3 = []
    skipped = 0

    for entry in tqdm(manifest, desc="Analyzing"):
        masked_path = entry["masked_path"]
        gt_label = entry.get("gt_label")

        try:
            image = Image.open(masked_path).convert("RGB")
            prediction = analyzer.analyze(image)

            if prediction is None:
                skipped += 1
                continue

            results_sam3.append({
                "image_path": entry["original_path"],
                "masked_path": masked_path,
                "gt_label": gt_label,
                "prediction": prediction,
                "mode": f'sam3_{entry.get("mask_type", "masked")}',
                "tumor_pixels": entry.get("tumor_pixels"),
                "total_pixels": entry.get("total_pixels"),
                "tumor_fraction": entry.get("tumor_fraction"),
                "dataset": entry.get("dataset"),
            })

        except Exception as e:
            print(f"\nWarning: skipping {masked_path}: {e}")
            skipped += 1
            continue

    print(f"\n[OK] Evaluated: {len(results_sam3)}  |  Skipped: {skipped}")

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_sam3_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_sam3, f, indent=2)
    print(f"[OK] Results saved: {results_path}")

    # Optionally load standalone baseline
    results_standalone = []
    if args.standalone_jsonl:
        results_standalone = load_standalone_from_jsonl(
            args.standalone_jsonl,
            split_filter=args.medgemma_split,
            tumor_only=not args.include_non_tumor_baseline,
        )
        print(f"[OK] Loaded {len(results_standalone)} standalone baseline rows from JSONL")

    # Compute and display metrics
    compute_metrics(results_standalone, results_sam3, output_dir)


# ============================================================================
# METRICS + PLOTS
# ============================================================================

def _extract_predictions(results):
    y_true, y_pred = [], []
    for r in results:
        gt = _norm_label(r.get("gt_label"))
        pred = _extract_prediction_label(r.get("prediction", {}))
        if gt and pred:
            y_true.append(gt)
            y_pred.append(pred)
    return y_true, y_pred


def compute_metrics(results_standalone, results_sam3, output_dir):
    print(f"\n{'='*80}\nRESULTS\n{'='*80}")

    metrics = {}

    if results_standalone:
        y_true, y_pred = _extract_predictions(results_standalone)
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["standalone"] = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                                         y_true=y_true, y_pred=y_pred)
            print(f"\nStandalone MedGemma (baseline):")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

    if results_sam3:
        y_true, y_pred = _extract_predictions(results_sam3)
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["sam3"] = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                                   y_true=y_true, y_pred=y_pred)
            print(f"\nSAM3 + MedGemma:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

    if "standalone" in metrics and "sam3" in metrics:
        acc_s, acc_sam = metrics["standalone"]["acc"], metrics["sam3"]["acc"]
        f1_s, f1_sam   = metrics["standalone"]["f1"],  metrics["sam3"]["f1"]
        acc_pct = ((acc_sam / acc_s - 1) * 100) if acc_s > 0 else float("nan")
        f1_pct  = ((f1_sam  / f1_s  - 1) * 100) if f1_s  > 0 else float("nan")
        print(f"\nDelta (SAM3 vs Standalone):")
        print(f"  Accuracy: {(acc_sam - acc_s):+.4f} ({acc_pct:+.1f}%)")
        print(f"  F1-Score: {(f1_sam - f1_s):+.4f}  ({f1_pct:+.1f}%)")

    print("="*80)

    if not metrics:
        print("No metrics to plot.")
        return

    _plot_results(metrics, output_dir)


def _plot_results(metrics, output_dir):
    has_both = "standalone" in metrics and "sam3" in metrics
    ncols = 2 if has_both else 1
    fig, axes = plt.subplots(1, 1 + (1 if "sam3" in metrics else 0), figsize=(7 * ncols, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax_idx = 0

    # Bar chart (only meaningful when comparing)
    if has_both:
        ax = axes[ax_idx]; ax_idx += 1
        labels = ["Standalone\nMedGemma", "SAM3 +\nMedGemma"]
        accs = [metrics["standalone"]["acc"], metrics["sam3"]["acc"]]
        bars = ax.bar(labels, accs, color=["#1f77b4", "#ff7f0e"], width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison")
        ax.axhline(y=0.6, color="red", linestyle="--", alpha=0.3, label="0.60 baseline")
        ax.legend(fontsize=8)
        for i, acc in enumerate(accs):
            ax.text(i, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    # Confusion matrix for SAM3 results
    if "sam3" in metrics:
        ax = axes[ax_idx]
        y_true = metrics["sam3"]["y_true"]
        y_pred = metrics["sam3"]["y_pred"]
        labels_uniq = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_uniq)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels_uniq, yticklabels=labels_uniq)
        ax.set_title("SAM3 + MedGemma — Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "evaluation_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Plot saved: {plot_path}")
    plt.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: MedGemma Evaluation")
    parser.add_argument("--manifest", required=True,
                        help="Path to segmentation_manifest_<timestamp>.json from Phase 1")
    parser.add_argument("--output_dir", default="./results/tumor_evaluation",
                        help="Where to save results and plots")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="vLLM OpenAI-compatible endpoint URL")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Model name to pass to the API")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of images (for testing)")
    # Optional standalone baseline from a prior JSONL run
    parser.add_argument("--standalone_jsonl", nargs="+", default=None,
                        help="MedGemma JSONL baseline files for comparison")
    parser.add_argument("--medgemma_split", default=None,
                        help="Optional split filter for JSONL baseline")
    parser.add_argument("--include_non_tumor_baseline", action="store_true",
                        help="Do not filter JSONL baseline to tumor rows only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
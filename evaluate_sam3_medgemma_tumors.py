#!/usr/bin/env python3
"""
SAM3 + MedGemma Evaluation Script for Brain Tumors

Experiments:
1. Standalone MedGemma (baseline)
2. SAM3 Segmentation + MedGemma (with soft masking)

Datasets:
- Brain Tumor Detection (Kaggle)
- Brain Tumor MRI 17 Classes (Kaggle)
- Brain Tumor MRI 44 Classes (Kaggle)
- Figshare Brain Tumor Dataset

Usage:
    python evaluate_sam3_medgemma_tumors.py \
        --data_dirs ./tumor_datasets \
        --sam3_checkpoint ./checkpoints/final_probe.pth \
        --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --output_dir ./results/tumor_evaluation \
        --mode both  # standalone, sam3, or both
"""

import os
import json
import argparse
import glob
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# ============================================================================
# CONFIGURATION
# ============================================================================

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://medgemma.openbrain.io/v1/chat/completions")
MODEL_NAME = "google/medgemma-1.5-4b-it"


def _resolve_device(requested="auto"):
    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return req


def _normalize_probe_state_dict(ckpt):
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format for linear probe")

    if "weight" in state and "bias" in state:
        return {"weight": state["weight"], "bias": state["bias"]}
    if "classifier.weight" in state and "classifier.bias" in state:
        return {"weight": state["classifier.weight"], "bias": state["classifier.bias"]}
    if "module.classifier.weight" in state and "module.classifier.bias" in state:
        return {"weight": state["module.classifier.weight"], "bias": state["module.classifier.bias"]}

    raise ValueError(f"Unsupported probe checkpoint keys: {list(state.keys())[:5]}")


def _norm_label(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "_")
    else:
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


TUMOR_DATASETS = {"images-17", "images-44c", "figshare", "br35h"}


def _is_tumor_dataset(dataset_name):
    if dataset_name is None:
        return False
    return str(dataset_name).strip().lower() in TUMOR_DATASETS


def _expand_jsonl_paths(jsonl_paths):
    expanded = []
    seen = set()
    for raw_path in jsonl_paths or []:
        if not raw_path:
            continue
        candidates = []
        if os.path.isdir(raw_path):
            candidates = sorted(glob.glob(os.path.join(raw_path, "*summary*.jsonl")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(raw_path, "*.jsonl")))
        elif any(token in raw_path for token in ["*", "?", "["]):
            candidates = sorted(glob.glob(raw_path))
        else:
            candidates = [raw_path]

        for p in candidates:
            if p not in seen:
                expanded.append(p)
                seen.add(p)
    return expanded


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
        chunks = [c.strip() for c in text.split("```") if c.strip()]
        for chunk in chunks:
            candidate = chunk[4:].strip() if chunk.startswith("json") else chunk
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def load_standalone_from_jsonl(jsonl_paths, split_filter=None, tumor_only=True):
    rows = []
    expanded_paths = _expand_jsonl_paths(jsonl_paths)
    if not expanded_paths:
        return rows

    for path in expanded_paths:
        if not os.path.exists(path):
            print(f"Warning: JSONL file not found: {path}")
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
                gt_name = _norm_label(meta.get("class"))
                gt_subclass = _norm_label(meta.get("subclass"))
                gt_label = gt_subclass if gt_subclass else gt_name

                if tumor_only and not _is_tumor_dataset(meta.get("dataset")):
                    continue

                output = obj.get("output", {}) or {}
                parsed = output.get("parsed_response", {}) or {}
                if output.get("status") != "success" or not isinstance(parsed, dict) or not parsed:
                    continue

                rows.append({
                    "image_path": obj.get("input", {}).get("image_path"),
                    "gt_label": gt_label,
                    "prediction": parsed,
                    "mode": "standalone_jsonl",
                    "source_file": path,
                    "split": processing.get("split"),
                })
    return rows




# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def apply_soft_mask(image, mask, background_dim=0.3):
    """
    Apply soft mask to image (dim background, keep foreground)
    
    Args:
        image: numpy array [H, W, 3] or PIL Image
        mask: binary mask [H, W]
        background_dim: float 0-1, how much to dim background
        
    Returns:
        masked_image: numpy array [H, W, 3] uint8
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure float
    image = image.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3c = np.expand_dims(mask, axis=-1)
    
    # Dim background
    background = image * (1 - mask_3c) * background_dim
    
    # Keep foreground strong
    foreground = image * mask_3c
    
    # Blend
    blended = foreground + background
    
    return (blended * 255).astype(np.uint8)


def apply_bounding_box(image, mask, color=(255, 0, 0), thickness=3):
    """
    Draw bounding box around tumor region
    
    Args:
        image: PIL Image
        mask: binary mask [H, W]
        color: RGB tuple for box color
        thickness: line thickness
        
    Returns:
        image_with_box: PIL Image
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        # No tumor found
        return image
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Draw box
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    
    return image


def apply_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Apply colored overlay on tumor region
    
    Args:
        image: PIL Image or numpy array
        mask: binary mask [H, W]
        color: RGB tuple for overlay
        alpha: transparency 0-1
        
    Returns:
        overlayed_image: numpy array [H, W, 3] uint8
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    
    # Blend
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    return blended.astype(np.uint8)


# ============================================================================
# MEDGEMMA ANALYZER
# ============================================================================

class MedGemmaAnalyzer:
    """MedGemma API wrapper"""
    
    def __init__(self, endpoint=VLLM_ENDPOINT, model=MODEL_NAME):
        self.endpoint = endpoint
        self.model = model
    
    def encode_image(self, image):
        """Convert PIL Image to base64"""
        import base64
        from io import BytesIO
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def analyze(self, image, system_prompt=SYSTEM_PROMPT, timeout=30):
        """
        Analyze image with MedGemma
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Parsed JSON response or None on error
        """
        image_b64 = self.encode_image(image)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON (supports fenced or wrapped JSON)
            parsed = _parse_json_payload(content)
            if parsed is None:
                raise ValueError("Failed to parse model JSON output")
            return parsed
            
        except Exception as e:
            print(f"Warning MedGemma error: {e}")
            return None


# ============================================================================
# DATASET LOADER
# ============================================================================

class TumorDatasetLoader:
    """Load tumor images from multiple datasets"""
    
    def __init__(self, data_dirs):
        """
        Args:
            data_dirs: list of dataset directories
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.samples = []
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Scan all directories for images"""
        
        print(f"\n{'='*80}")
        print("LOADING DATASETS")
        print('='*80)
        
        for data_dir in self.data_dirs:
            dataset_name = Path(data_dir).name
            print(f"\nScanning: {dataset_name}")
            
            # Find all images
            image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for ext in image_exts:
                image_files.extend(Path(data_dir).rglob(ext))
            
            print(f"  Found {len(image_files)} images")
            
            # Determine ground truth from directory structure
            for img_path in image_files:
                # Try to extract label from path
                label = self._extract_label(img_path)
                
                self.samples.append({
                    'image_path': str(img_path),
                    'dataset': dataset_name,
                    'gt_label': label
                })
        
        print(f"\n[OK] Total samples: {len(self.samples)}")
        print('='*80)
    
    def _extract_label(self, img_path):
        """Extract ground truth label from file path"""
        path_str = str(img_path).lower()
        
        # Common tumor class names
        if any(x in path_str for x in ['glioma', 'astrocytoma', 'glioblastoma']):
            return 'glioma'
        elif 'meningioma' in path_str:
            return 'meningioma'
        elif 'pituitary' in path_str:
            return 'pituitary_tumor'
        elif any(x in path_str for x in ['normal', 'no_tumor', 'notumor']):
            return 'normal'
        else:
            # Check parent directory
            parent = Path(img_path).parent.name.lower()
            if 'glioma' in parent:
                return 'glioma'
            elif 'meningioma' in parent:
                return 'meningioma'
            elif 'pituitary' in parent:
                return 'pituitary_tumor'
            elif 'normal' in parent or 'no' in parent:
                return 'normal'
            else:
                return 'tumor'  # Generic tumor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class SAM3MedGemmaEvaluator:
    """Complete evaluation pipeline"""
    
    def __init__(self, sam3_checkpoint, bpe_path, output_dir, device='auto'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.segmentor = SAM3Segmentor(sam3_checkpoint, bpe_path, device)
        self.analyzer = MedGemmaAnalyzer()
        self.device = device
    
    def run_experiment(self, dataset, mode='both', mask_type='soft', 
                      background_dim=0.3, max_samples=None):
        """
        Run evaluation experiment
        
        Args:
            dataset: TumorDatasetLoader
            mode: 'standalone', 'sam3', or 'both'
            mask_type: 'soft', 'bbox', or 'overlay'
            background_dim: for soft mask
            max_samples: limit number of samples (for testing)
        """
        print(f"\n{'='*80}")
        print("RUNNING EXPERIMENT")
        print('='*80)
        print(f"Mode: {mode}")
        print(f"Mask type: {mask_type}")
        print(f"Samples: {len(dataset) if max_samples is None else max_samples}")
        print('='*80 + '\n')
        
        results_standalone = []
        results_sam3 = []
        
        samples = dataset.samples[:max_samples] if max_samples else dataset.samples
        
        for sample in tqdm(samples, desc="Processing"):
            img_path = sample['image_path']
            gt_label = sample['gt_label']
            
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Resize if too large
                max_size = 512
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Standalone MedGemma
                if mode in ['standalone', 'both']:
                    result_standalone = self.analyzer.analyze(image)
                    
                    if result_standalone:
                        results_standalone.append({
                            'image_path': img_path,
                            'gt_label': gt_label,
                            'prediction': result_standalone,
                            'mode': 'standalone'
                        })
                
                # SAM3 + MedGemma
                if mode in ['sam3', 'both']:
                    # Segment tumor
                    mask = self.segmentor.segment(image)
                    
                    # Apply mask/box
                    if mask_type == 'soft':
                        masked_image = apply_soft_mask(image, mask, background_dim)
                        masked_image = Image.fromarray(masked_image)
                    elif mask_type == 'bbox':
                        masked_image = apply_bounding_box(image, mask)
                    elif mask_type == 'overlay':
                        masked_image = apply_overlay(image, mask)
                        masked_image = Image.fromarray(masked_image)
                    else:
                        masked_image = image
                    
                    # Analyze masked image
                    result_sam3 = self.analyzer.analyze(masked_image)
                    
                    if result_sam3:
                        results_sam3.append({
                            'image_path': img_path,
                            'gt_label': gt_label,
                            'prediction': result_sam3,
                            'mode': f'sam3_{mask_type}',
                            'tumor_pixels': int(mask.sum()),
                            'total_pixels': int(mask.size)
                        })
            
            except Exception as e:
                print(f"\nWarning Error processing {img_path}: {e}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if results_standalone:
            standalone_file = self.output_dir / f'results_standalone_{timestamp}.json'
            with open(standalone_file, 'w') as f:
                json.dump(results_standalone, f, indent=2)
            print(f"\n[OK] Standalone results: {standalone_file}")
        
        if results_sam3:
            sam3_file = self.output_dir / f'results_sam3_{mask_type}_{timestamp}.json'
            with open(sam3_file, 'w') as f:
                json.dump(results_sam3, f, indent=2)
            print(f"[OK] SAM3 results: {sam3_file}")
        
        return results_standalone, results_sam3
    
    def analyze_results(self, results_standalone, results_sam3):
        """Compute metrics and generate comparison plots"""
        
        print(f"\n{'='*80}")
        print("RESULTS ANALYSIS")
        print('='*80)
        
        # Extract predictions
        def extract_predictions(results):
            y_true = []
            y_pred = []
            
            for r in results:
                gt = _norm_label(r.get('gt_label'))
                pred = _extract_prediction_label(r.get('prediction', {}))
                if gt is None or pred is None:
                    continue
                
                y_true.append(gt)
                y_pred.append(pred)
            
            return y_true, y_pred
        
        metrics_rows = []
        metrics_summary = {"generated_at": datetime.now().isoformat()}

        # Standalone metrics
        if results_standalone:
            y_true_s, y_pred_s = extract_predictions(results_standalone)
            acc_s = accuracy_score(y_true_s, y_pred_s)
            prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
                y_true_s, y_pred_s, average='weighted', zero_division=0
            )
            
            print("\nStandalone MedGemma:")
            print(f"  Accuracy:  {acc_s:.4f}")
            print(f"  Precision: {prec_s:.4f}")
            print(f"  Recall:    {rec_s:.4f}")
            print(f"  F1-Score:  {f1_s:.4f}")
            metrics_summary["standalone"] = {
                "samples": len(y_true_s),
                "accuracy": float(acc_s),
                "precision_weighted": float(prec_s),
                "recall_weighted": float(rec_s),
                "f1_weighted": float(f1_s),
            }
            metrics_rows.append({
                "method": "standalone",
                "samples": len(y_true_s),
                "accuracy": float(acc_s),
                "precision_weighted": float(prec_s),
                "recall_weighted": float(rec_s),
                "f1_weighted": float(f1_s),
            })
        
        # SAM3 metrics
        if results_sam3:
            y_true_sam, y_pred_sam = extract_predictions(results_sam3)
            acc_sam = accuracy_score(y_true_sam, y_pred_sam)
            prec_sam, rec_sam, f1_sam, _ = precision_recall_fscore_support(
                y_true_sam, y_pred_sam, average='weighted', zero_division=0
            )
            
            print("\nSAM3 + MedGemma:")
            print(f"  Accuracy:  {acc_sam:.4f}")
            print(f"  Precision: {prec_sam:.4f}")
            print(f"  Recall:    {rec_sam:.4f}")
            print(f"  F1-Score:  {f1_sam:.4f}")
            metrics_summary["sam3"] = {
                "samples": len(y_true_sam),
                "accuracy": float(acc_sam),
                "precision_weighted": float(prec_sam),
                "recall_weighted": float(rec_sam),
                "f1_weighted": float(f1_sam),
            }
            metrics_rows.append({
                "method": "sam3",
                "samples": len(y_true_sam),
                "accuracy": float(acc_sam),
                "precision_weighted": float(prec_sam),
                "recall_weighted": float(rec_sam),
                "f1_weighted": float(f1_sam),
            })
        
        # Comparison
        if results_standalone and results_sam3:
            print(f"\nImprovement:")
            acc_pct = ((acc_sam / acc_s - 1) * 100) if acc_s > 0 else float("nan")
            f1_pct = ((f1_sam / f1_s - 1) * 100) if f1_s > 0 else float("nan")
            print(f"  Accuracy:  {(acc_sam - acc_s):+.4f} ({acc_pct:+.1f}%)")
            print(f"  F1-Score:  {(f1_sam - f1_s):+.4f} ({f1_pct:+.1f}%)")
            metrics_summary["improvement"] = {
                "accuracy_delta": float(acc_sam - acc_s),
                "accuracy_pct": float(acc_pct),
                "f1_delta": float(f1_sam - f1_s),
                "f1_pct": float(f1_pct),
            }
        
        print('='*80)

        metrics_json_path = self.output_dir / "metrics_summary.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"[OK] Metrics saved: {metrics_json_path}")

        if metrics_rows:
            metrics_csv_path = self.output_dir / "metrics_summary.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)
            print(f"[OK] Metrics table: {metrics_csv_path}")
        
        # Generate plots
        self._plot_comparison(results_standalone, results_sam3)
    
    def _plot_comparison(self, results_standalone, results_sam3):
        """Generate comparison plots"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        if results_standalone and results_sam3:
            y_true_s, y_pred_s = self._extract_predictions(results_standalone)
            y_true_sam, y_pred_sam = self._extract_predictions(results_sam3)
            
            acc_s = accuracy_score(y_true_s, y_pred_s)
            acc_sam = accuracy_score(y_true_sam, y_pred_sam)
            
            methods = ['Standalone\nMedGemma', 'SAM3 +\nMedGemma']
            accuracies = [acc_s, acc_sam]
            
            bars = axes[0].bar(methods, accuracies, color=['#1f77b4', '#ff7f0e'])
            axes[0].set_ylim(0, 1)
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Accuracy Comparison')
            axes[0].axhline(y=0.6, color='red', linestyle='--', alpha=0.3)
            
            # Add value labels
            for i, (method, acc) in enumerate(zip(methods, accuracies)):
                axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for SAM3
        if results_sam3:
            y_true, y_pred = self._extract_predictions(results_sam3)
            cm = confusion_matrix(y_true, y_pred)
            
            # Get unique labels
            labels = sorted(set(y_true + y_pred))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       xticklabels=labels, yticklabels=labels)
            axes[1].set_title('SAM3 + MedGemma Confusion Matrix')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Plot saved: {plot_path}")
        plt.close()
    
    def _extract_predictions(self, results):
        """Helper to extract predictions"""
        y_true = []
        y_pred = []
        
        for r in results:
            gt = _norm_label(r.get('gt_label'))
            pred = _extract_prediction_label(r.get('prediction', {}))
            if gt is None or pred is None:
                continue
            
            y_true.append(gt)
            y_pred.append(pred)
        
        return y_true, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    # New flow: optionally reuse standalone JSONL tumor baseline.
    results_standalone = []
    results_sam3 = []
    use_jsonl_baseline = bool(getattr(args, "standalone_jsonl", None))

    if use_jsonl_baseline:
        results_standalone = load_standalone_from_jsonl(
            args.standalone_jsonl,
            split_filter=getattr(args, "medgemma_split", None),
            tumor_only=not getattr(args, "include_non_tumor_baseline", False),
        )
        print(f"Loaded {len(results_standalone)} baseline rows from JSONL.")

    need_sam3 = args.mode in ["sam3", "both"] or (args.mode == "standalone" and not use_jsonl_baseline)
    evaluator = None

    if need_sam3:
        if not args.sam3_checkpoint or not args.bpe_path:
            raise ValueError("--sam3_checkpoint and --bpe_path are required when SAM3 inference is used")
        evaluator = SAM3MedGemmaEvaluator(
            sam3_checkpoint=args.sam3_checkpoint,
            bpe_path=args.bpe_path,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        # Baseline-only mode: avoid loading SAM3 weights.
        evaluator = SAM3MedGemmaEvaluator.__new__(SAM3MedGemmaEvaluator)
        evaluator.output_dir = Path(args.output_dir)
        evaluator.output_dir.mkdir(parents=True, exist_ok=True)

    if need_sam3:
        if not args.data_dirs:
            raise ValueError(
                "--data_dirs is required unless you pass --standalone_jsonl with --mode standalone"
            )
        dataset = TumorDatasetLoader(args.data_dirs)
        mode_for_run = "sam3" if (use_jsonl_baseline and args.mode == "both") else args.mode
        rs, r3 = evaluator.run_experiment(
            dataset=dataset,
            mode=mode_for_run,
            mask_type=args.mask_type,
            background_dim=args.background_dim,
            max_samples=args.max_samples
        )
        if not use_jsonl_baseline:
            results_standalone = rs
        results_sam3 = r3

    if results_standalone or results_sam3:
        evaluator.analyze_results(results_standalone, results_sam3)

    print("\nEvaluation complete!")
    return

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3 + MedGemma Tumor Evaluation')
    
    parser.add_argument('--data_dirs', nargs='+', default=None,
                        help='Directories containing tumor datasets')
    parser.add_argument('--sam3_checkpoint', default=None,
                        help='Path to SAM3 linear probe checkpoint')
    parser.add_argument('--bpe_path', default=None,
                        help='Path to BPE vocabulary file')
    parser.add_argument('--output_dir', default='./results/tumor_eval',
                        help='Output directory for results')
    parser.add_argument('--mode', choices=['standalone', 'sam3', 'both'], default='both',
                        help='Evaluation mode')
    parser.add_argument('--mask_type', choices=['soft', 'bbox', 'overlay'], default='soft',
                        help='Type of mask to apply')
    parser.add_argument('--background_dim', type=float, default=0.3,
                        help='Background dimming factor for soft mask (0-1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to process (for testing)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device for SAM3 inference')
    parser.add_argument('--standalone_jsonl', nargs='+', default=None,
                        help='Optional MedGemma JSONL baseline inputs (files, globs, or directories)')
    parser.add_argument('--medgemma_split', default=None,
                        help='Optional split filter for JSONL baseline (e.g., subset_1, subset_0_25_7)')
    parser.add_argument('--include_non_tumor_baseline', action='store_true',
                        help='Do not filter JSONL baseline to tumor rows only')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

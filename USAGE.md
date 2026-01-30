# Linear Probe Usage Guide

## File Structure

```
SAM 3/
├── models.py              # Model architectures (LinearProbe class)
├── linear_probing.py      # Training script
├── inference.py           # Inference script
└── checkpoints/
    ├── final_probe.pth    # Saved model weights (state_dict)
    ├── train_features.h5  # Pre-computed training features
    └── val_features.h5    # Pre-computed validation features
```

## Training

```bash
python linear_probing.py \
    --data_root /path/to/BraTS2020 \
    --bpe_path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --save_dir ./checkpoints \
    --batch_size 64 \
    --epochs 20 \
    --lr 0.001
```

## Loading the Trained Model

The model is saved as a `state_dict` (weights only). To load it:

```python
import torch
from models import LinearProbe

# Initialize model architecture
model = LinearProbe(feature_dim=256, num_classes=2)

# Load trained weights
model.load_state_dict(torch.load('checkpoints/final_probe.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    logits = model(features)  # features: [B, 256, H, W]
    predictions = torch.argmax(logits, dim=1)
```

## Inference

Run inference on pre-computed features:

```bash
python inference.py \
    --checkpoint ./checkpoints/final_probe.pth \
    --features_path ./checkpoints/val_features.h5 \
    --feature_dim 256 \
    --batch_size 32
```

## Important Notes

1. **Feature dimension**: Must match what was used during training (check first feature in .h5 file)
2. **models.py required**: The `LinearProbe` class definition is needed to load weights
3. **HDF5 features**: Pre-computed features stay on disk, loaded batch-by-batch (memory efficient)

## Checking Feature Dimension

```python
import h5py

with h5py.File('checkpoints/val_features.h5', 'r') as f:
    feature_shape = f['features'][0].shape
    print(f"Feature shape: {feature_shape}")
    print(f"Feature dim: {feature_shape[0]}")
```

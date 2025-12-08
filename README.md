# Red Light Violation Detection

Production-ready system for detecting vehicles, traffic lights, and stop lines using YOLOv8. This forms the foundation for red light violation detection logic.

## Project Structure

```
red-light/
├── data/                          # Dataset (YOLOv8 format)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml                  # Dataset configuration
├── configs/                       # Training configurations
│   ├── baseline.yaml              # Fast baseline (YOLOv8n)
│   ├── enhanced.yaml              # Better accuracy (YOLOv8s)
│   └── production.yaml            # Best accuracy (YOLOv8m)
├── experiments/                   # Training outputs
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── data_exploration.ipynb         # Dataset analysis notebook
└── requirements.txt               # Python dependencies
```

## Dataset Overview

**Classes** (9 total):
- **Vehicles**: bike, bus, car, motobike, truck
- **Traffic Lights**: red_light, green_light, yellow_light
- **Infrastructure**: stop_line

**Statistics**:
- Total Images: 3,395
- Total Annotations: 15,619
- Train: ~2,200 images | Valid: ~680 images | Test: ~340 images
- Average 4.5 objects per image

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully!')"
```

## Quick Start

### 1. Explore the Dataset

```bash
jupyter notebook data_exploration.ipynb
```

This notebook provides:
- Dataset statistics and class distribution
- Bounding box analysis
- Sample visualizations with annotations
- Class imbalance analysis

### 2. Train a Model

#### Option A: Baseline (Fast - Recommended for First Run)

```bash
python train.py --config configs/baseline.yaml
```

- **Model**: YOLOv8 Nano (fastest)
- **Training Time**: ~1-2 hours on GPU
- **Use Case**: Quick baseline, testing pipeline

#### Option B: Enhanced (Better Accuracy)

```bash
python train.py --config configs/enhanced.yaml
```

- **Model**: YOLOv8 Small
- **Training Time**: ~3-4 hours on GPU
- **Use Case**: Better accuracy with enhanced augmentation

#### Option C: Production (Best Accuracy)

```bash
python train.py --config configs/production.yaml
```

- **Model**: YOLOv8 Medium
- **Training Time**: ~6-8 hours on GPU
- **Use Case**: Production deployment

### 3. Monitor Training

Training outputs are saved to `experiments/[experiment_name]_[timestamp]/`:

```
experiments/baseline_yolov8n_20231208_143022/
├── runs/train/
│   ├── weights/
│   │   ├── best.pt              # Best model weights
│   │   └── last.pt              # Last epoch weights
│   ├── results.csv              # Training metrics
│   ├── results.png              # Training curves
│   └── confusion_matrix.png     # Confusion matrix
├── config.yaml                  # Training configuration
└── training_summary.json        # Training metadata
```

### 4. Evaluate Model

```bash
# Evaluate on validation set
python evaluate.py --model experiments/[exp_dir]/runs/train/weights/best.pt --split val

# Evaluate on test set
python evaluate.py --model experiments/[exp_dir]/runs/train/weights/best.pt --split test
```

Evaluation outputs:
- Per-class Average Precision (AP@0.5, AP@0.5:0.95)
- Performance visualizations
- Classes needing attention
- Detailed metrics JSON

## Training Configuration

All training parameters are configured via YAML files in `configs/`. Key parameters:

### Model Selection
- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (more accurate)
- `yolov8l.pt` - Large (most accurate, slowest)
- `yolov8x.pt` - Extra Large (best accuracy)

### Training Parameters
```yaml
epochs: 100              # Number of training epochs
batch_size: 16          # Batch size (adjust for GPU memory)
imgsz: 640              # Input image size
device: 0               # GPU ID (or 'cpu')
patience: 50            # Early stopping patience
```

### Augmentation
```yaml
augmentation:
  hsv_h: 0.02           # Hue variation (traffic lights)
  hsv_s: 0.8            # Saturation variation
  hsv_v: 0.5            # Brightness variation
  fliplr: 0.5           # Horizontal flip probability
  mosaic: 1.0           # Mosaic augmentation
  mixup: 0.1            # MixUp for rare classes
```

## Custom Configuration

Create a custom config file:

```yaml
# configs/custom.yaml
experiment_name: my_experiment
model: yolov8s.pt
data_yaml: data/data.yaml
epochs: 150
batch_size: 16
imgsz: 640
# ... add other parameters
```

Run training:

```bash
python train.py --config configs/custom.yaml
```

## GPU Requirements

Recommended GPU memory by model size:

| Model    | Parameters | GPU Memory | Batch Size |
|----------|-----------|------------|------------|
| YOLOv8n  | 3.2M      | 4GB+       | 32         |
| YOLOv8s  | 11.2M     | 6GB+       | 16         |
| YOLOv8m  | 25.9M     | 8GB+       | 12         |
| YOLOv8l  | 43.7M     | 12GB+      | 8          |
| YOLOv8x  | 68.2M     | 16GB+      | 4          |

**No GPU?** Training will run on CPU (slower). Set `device: cpu` in config.

## Handling Class Imbalance

The dataset has class imbalance (motobike: 4,116 vs bus: 389). Our configs handle this through:

1. **Enhanced Augmentation**: Mixup and copy-paste for rare classes
2. **Longer Training**: More epochs for better convergence
3. **Per-Class Monitoring**: Evaluation script tracks all classes

If specific classes underperform after training:
1. Check evaluation output for "Classes Needing Attention"
2. Increase `mixup` and `copy_paste` augmentation
3. Adjust `cls` loss weight in hyperparameters
4. Consider collecting more data for rare classes

## Inference (Coming Soon)

After training, you can use the model for inference:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('experiments/[exp_dir]/runs/train/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
```

## Next Steps

After training the detection model:

1. **Evaluate Performance**: Use `evaluate.py` to analyze results
2. **Implement Violation Logic**: Build spatial relationship logic to detect when vehicles cross stop lines during red lights
3. **Add Object Tracking**: Track vehicles across frames for temporal analysis
4. **Deploy Model**: Export to ONNX/TensorRT for production inference

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config
- Use smaller model (nano/small instead of medium/large)
- Enable `cache: false` if using cache

### Slow Training
- Enable `amp: true` for mixed precision training
- Use smaller model for faster experiments
- Increase `workers` for data loading (if CPU allows)

### Poor Performance on Specific Classes
- Check evaluation report for low-performing classes
- Increase augmentation for those classes
- Consider collecting more training data
- Adjust class loss weight in hyperparameters

### Training Stops Early
- Increase `patience` parameter
- Check if validation loss is actually not improving
- Try different learning rate (`lr0` in hyperparameters)

## Support

For issues or questions:
1. Check training logs in `experiments/[exp_dir]/runs/train/`
2. Review evaluation metrics and visualizations
3. Consult YOLOv8 documentation: https://docs.ultralytics.com/

## License

Dataset: CC BY 4.0 (from Roboflow)

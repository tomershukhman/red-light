# Quick Start Guide

Get your red light detection model training in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Train Your First Model (1-2 hours)

```bash
# Start with the baseline (fastest option)
python train.py --config configs/baseline.yaml
```

This will:
- Train YOLOv8 Nano model
- Use default augmentation settings
- Save results to `experiments/baseline_yolov8n_[timestamp]/`
- Take ~1-2 hours on GPU (or 8-12 hours on CPU)

**Monitor progress**: Check `experiments/*/runs/train/results.png` for training curves

## Step 3: Evaluate Your Model (1 minute)

```bash
# Replace [exp_dir] with your experiment directory name
python evaluate.py \
  --model experiments/[exp_dir]/runs/train/weights/best.pt \
  --split val
```

This will show:
- Overall mAP@0.5 and mAP@0.5:0.95
- Per-class performance metrics
- Classes that need attention
- Save visualizations to `experiments/[exp_dir]/evaluation/`

## Step 4: Test Inference (30 seconds)

```bash
# Test on a single image
python infer.py \
  --model experiments/[exp_dir]/runs/train/weights/best.pt \
  --source data/test/images/[any_image].jpg \
  --conf 0.25

# Or test on entire directory
python infer.py \
  --model experiments/[exp_dir]/runs/train/weights/best.pt \
  --source data/test/images/ \
  --conf 0.25
```

Results saved to `inference_results/`

## Common Commands

### Train Different Models

```bash
# Fast baseline (YOLOv8n) - ~1-2h on GPU
python train.py --config configs/baseline.yaml

# Better accuracy (YOLOv8s) - ~3-4h on GPU
python train.py --config configs/enhanced.yaml

# Best accuracy (YOLOv8m) - ~6-8h on GPU
python train.py --config configs/production.yaml
```

### Evaluate on Test Set

```bash
python evaluate.py \
  --model experiments/[exp_dir]/runs/train/weights/best.pt \
  --split test
```

### Inference with Lower/Higher Confidence

```bash
# Lower confidence (more detections)
python infer.py --model path/to/best.pt --source image.jpg --conf 0.1

# Higher confidence (fewer, more certain detections)
python infer.py --model path/to/best.pt --source image.jpg --conf 0.5
```

## Expected Results

Based on your dataset analysis:

### What to Expect:
- **Overall mAP@0.5**: ~0.70-0.85 (depending on model size)
- **High-performing classes**: car, motobike, stop_line, red_light
- **May need attention**: bus, bike (fewer training samples)

### If Results Are Poor:
1. Check evaluation output for specific weak classes
2. Try enhanced config with more augmentation
3. Train longer (increase epochs)
4. Use larger model (s, m instead of n)

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size in config file
# Change: batch_size: 16
# To: batch_size: 8  (or smaller)
```

### "Training is too slow"
```bash
# Use smaller model
python train.py --config configs/baseline.yaml

# Or reduce image size in config
# Change: imgsz: 640
# To: imgsz: 416
```

### "No GPU detected"
```bash
# Training will run on CPU (slower but works)
# Set in config: device: cpu
```

## Next Steps

After successful training:

1. **Compare Models**: Train multiple configs and compare results
2. **Analyze Failures**: Look at images where model performs poorly
3. **Fine-tune**: Adjust augmentation/hyperparameters for weak classes
4. **Export Model**: Export to ONNX for production deployment
5. **Build Violation Logic**: Add spatial reasoning to detect red light violations

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review training logs: `experiments/*/runs/train/results.csv`
- Examine evaluation plots: `experiments/*/evaluation/*.png`
- Explore dataset: Open `data_exploration.ipynb` in Jupyter

## Pro Tips

1. **Always start with baseline**: Get quick results, identify issues early
2. **Monitor per-class metrics**: Some classes may need special attention
3. **Save your experiments**: Each run saves to separate timestamped directory
4. **Use GPU if available**: 10-20x faster than CPU training
5. **Experiment with augmentation**: Traffic scenarios benefit from realistic augmentation

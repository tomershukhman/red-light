#!/bin/bash
# Quick verification that WandB is properly configured for YOLO

echo "============================================================"
echo "WandB Integration Verification"
echo "============================================================"
echo ""

# Check if wandb is enabled in YOLO settings
echo "1. Checking YOLO settings..."
WANDB_ENABLED=$(yolo settings | grep -o '"wandb": true')
if [ -n "$WANDB_ENABLED" ]; then
    echo "   ✅ WandB is ENABLED in YOLO settings"
else
    echo "   ❌ WandB is DISABLED in YOLO settings"
    echo "   Run: yolo settings wandb=True"
    exit 1
fi

# Check if wandb package is installed
echo ""
echo "2. Checking wandb package..."
if python -c "import wandb" 2>/dev/null; then
    WANDB_VERSION=$(python -c "import wandb; print(wandb.__version__)")
    echo "   ✅ wandb package installed (version $WANDB_VERSION)"
else
    echo "   ❌ wandb package not installed"
    echo "   Run: pip install wandb"
    exit 1
fi

# Check if logged in to wandb
echo ""
echo "3. Checking WandB authentication..."
if wandb login --relogin 2>&1 | grep -q "Already logged in"; then
    echo "   ✅ Logged in to WandB"
elif [ -f ~/.netrc ] && grep -q "api.wandb.ai" ~/.netrc; then
    echo "   ✅ WandB credentials found"
else
    echo "   ⚠️  May need to login to WandB"
    echo "   Run: wandb login"
fi

echo ""
echo "============================================================"
echo "✅ All checks passed! Ready to train with WandB logging"
echo "============================================================"
echo ""
echo "Start training with:"
echo "  python train.py --config configs/train_yolov8s_aug.yaml"
echo ""


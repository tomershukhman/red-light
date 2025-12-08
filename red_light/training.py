"""
Training pipeline utilities for Red Light Violation Detection.

Expose `RedLightDetectionTrainer` for programmatic use; CLI wrappers should
stay thin and delegate here.
"""

from pathlib import Path
import json

from ultralytics import YOLO
import torch

from red_light.config import load_training_config
from red_light.experiment import ExperimentManager
from red_light.tracking import WandbTracker


class RedLightDetectionTrainer:
    """Production-ready trainer for red light violation detection model."""

    def __init__(self, config_path):
        """
        Initialize trainer with configuration.

        Args:
            config_path: Path to training configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = load_training_config(self.config_path)
        self.experiment = ExperimentManager(self.config, self.config_path)
        self.exp_dir = self.experiment.setup_experiment()
        self.tracker = WandbTracker(
            self.config.get('tracking', {}), self.exp_dir)

    def train(self):
        """Train the YOLO model."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        resolved_device = self._resolve_device(
            self.config.get('training', {}).get('device'))
        self.resolved_device = resolved_device

        self._print_config_summary(resolved_device)

        model_name = self.config['model']
        print(f"\nLoading model: {model_name}")

        if not Path(model_name).exists() and not model_name.startswith('yolov8'):
            raise FileNotFoundError(f"Model file not found: {model_name}")

        model = YOLO(model_name)

        # YOLO has built-in wandb integration that automatically detects
        # an active wandb run and logs comprehensive metrics including:
        # - Per-batch training losses
        # - Training/validation images with predictions
        # - Confusion matrices, PR curves, F1 curves
        # - Model architecture, hyperparameters
        # No custom callback needed - YOLO handles it all!

        train_args = self._build_training_args(resolved_device)
        
        # Configure WandB tracking (sets environment variables for YOLO's integration)
        if self.tracker.enabled:
            self.tracker.get_yolo_wandb_config()
        
        self.tracker.start(resolved_device, train_args, self.config)

        print("\nTraining arguments:")
        print(json.dumps({k: str(v) for k, v in train_args.items()}, indent=2))

        print("\n" + "=" * 60)
        print("Training started...")
        print("=" * 60 + "\n")

        try:
            results = model.train(**train_args)

            print("\n" + "=" * 60)
            print("Training completed!")
            print("=" * 60)

            summary, summary_path = self.experiment.save_training_summary(
                train_args)
            self.tracker.log_summary(summary)
            self.tracker.log_artifacts(summary_path)

            return results, model
        finally:
            self.tracker.finish()

    def _build_training_args(self, resolved_device):
        """Build training arguments from config - no hardcoded defaults."""
        train_config = self.config.get('training', {})

        train_args = {
            'data': self.config['data_yaml'],
            'epochs': train_config.get('epochs'),
            'imgsz': train_config.get('imgsz'),
            'batch': train_config.get('batch_size'),
            'device': resolved_device,
            'workers': train_config.get('workers'),
            'project': str(self.exp_dir / 'runs'),
            'name': 'train',
            'exist_ok': True,
            'pretrained': train_config.get('pretrained'),
            'optimizer': train_config.get('optimizer'),
            'verbose': train_config.get('verbose', True),
            'seed': train_config.get('seed'),
            'deterministic': train_config.get('deterministic'),
            'save': True,
            'save_period': train_config.get('save_period'),
            'cache': train_config.get('cache'),
            'patience': train_config.get('patience'),
            'amp': train_config.get('amp'),
        }

        train_args = {k: v for k, v in train_args.items() if v is not None}

        aug_config = self.config.get('augmentation', {})
        if aug_config:
            aug_args = {
                'hsv_h': aug_config.get('hsv_h'),
                'hsv_s': aug_config.get('hsv_s'),
                'hsv_v': aug_config.get('hsv_v'),
                'degrees': aug_config.get('degrees'),
                'translate': aug_config.get('translate'),
                'scale': aug_config.get('scale'),
                'shear': aug_config.get('shear'),
                'perspective': aug_config.get('perspective'),
                'flipud': aug_config.get('flipud'),
                'fliplr': aug_config.get('fliplr'),
                'mosaic': aug_config.get('mosaic'),
                'mixup': aug_config.get('mixup'),
                'copy_paste': aug_config.get('copy_paste'),
            }
            aug_args = {k: v for k, v in aug_args.items() if v is not None}
            train_args.update(aug_args)

        hyp_config = self.config.get('hyperparameters', {})
        if hyp_config:
            hyp_args = {
                'lr0': hyp_config.get('lr0'),
                'lrf': hyp_config.get('lrf'),
                'momentum': hyp_config.get('momentum'),
                'weight_decay': hyp_config.get('weight_decay'),
                'warmup_epochs': hyp_config.get('warmup_epochs'),
                'warmup_momentum': hyp_config.get('warmup_momentum'),
                'warmup_bias_lr': hyp_config.get('warmup_bias_lr'),
                'box': hyp_config.get('box'),
                'cls': hyp_config.get('cls'),
                'dfl': hyp_config.get('dfl'),
            }
            hyp_args = {k: v for k, v in hyp_args.items() if v is not None}
            train_args.update(hyp_args)

        return train_args

    def _resolve_device(self, device_config):
        """Resolve device from config with safe CPU fallback when CUDA/MPS unavailable."""
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(
            torch.backends, 'mps') and torch.backends.mps.is_available()

        if device_config is None or str(device_config).lower() == 'auto':
            if cuda_available:
                return 'auto'
            if mps_available:
                return 'mps'
            return 'cpu'

        device_str = str(device_config).lower()

        if device_str not in ['cpu', 'mps'] and not cuda_available:
            print(
                f"Warning: CUDA device '{device_config}' requested but no CUDA devices detected. "
                "Falling back to CPU."
            )
            return 'cpu'

        if device_str == 'mps' and not mps_available:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return 'cpu'

        return device_config

    def _print_config_summary(self, resolved_device):
        """Print configuration summary."""
        train_config = self.config.get('training', {})

        print("\nConfiguration Summary:")
        print("-" * 60)
        print(f"Experiment Name: {self.config['experiment_name']}")
        print(f"Model: {self.config['model']}")
        print(f"Data YAML: {self.config['data_yaml']}")
        print(f"Epochs: {train_config.get('epochs', 'default')}")
        print(f"Batch Size: {train_config.get('batch_size', 'default')}")
        print(f"Image Size: {train_config.get('imgsz', 'default')}")
        print(f"Device: {resolved_device}")

        if torch.cuda.is_available():
            device_id = 0 if resolved_device in [
                'auto', 0, '0'] else resolved_device
            if isinstance(device_id, int) or (isinstance(device_id, str) and device_id.isdigit()):
                device_id = int(device_id) if isinstance(
                    device_id, str) else device_id
                print(
                    f"GPU Available: Yes ({torch.cuda.get_device_name(device_id)})")
                print(
                    f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
            else:
                print(f"GPU Available: Yes ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("GPU Available: MPS")
        else:
            print("GPU Available: No (training on CPU)")

        print("-" * 60)

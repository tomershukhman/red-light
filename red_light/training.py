"""
Training pipeline utilities for Red Light Violation Detection.

Simple wrapper around YOLO training with WandB integration.
"""

from pathlib import Path
import json
from datetime import datetime

from ultralytics import YOLO
import torch

from red_light.config import load_training_config
from red_light.tracking import setup_wandb


class RedLightDetectionTrainer:
    """Production-ready trainer for red light violation detection model."""

    def __init__(self, config_path):
        """Initialize trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = load_training_config(self.config_path)
        self._setup_experiment()

    def train(self):
        """Train the YOLO model with YOLO's built-in WandB integration."""
        self._print_header("Starting Training")
        
        device = self._resolve_device()
        self._print_config_summary(device)
        
        setup_wandb(self.config)
        model = self._load_model()
        train_args = self._build_training_args(device)
        
        self._print_training_args(train_args)
        self._print_header("Training started...")
        
        results = model.train(**train_args)
        
        self._print_header("Training completed!")
        print(f"Results saved in: {results.save_dir}")
        
        return results, model

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def _setup_experiment(self):
        """Setup experiment name and output directory."""
        # Add timestamp for unique experiment names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.config['experiment_name']
        self.config['experiment_name'] = f"{base_name}_{timestamp}"
        
        # Create output directory
        exp_config = self.config.get('experiment', {})
        self.project_dir = Path(exp_config.get('output_dir', 'runs'))
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Project directory: {self.project_dir}")
        print(f"Experiment name: {self.config['experiment_name']}")

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_model(self):
        """Load YOLO model (new or resume from checkpoint)."""
        resume_from = self.config.get('resume_from')
        
        if resume_from:
            return self._load_checkpoint(resume_from)
        
        return self._load_new_model()

    def _load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint to resume training."""
        self._print_header(f"RESUMING from: {checkpoint_path}")
        # Path validation already done in config.py
        print("Checkpoint loaded successfully")
        return YOLO(checkpoint_path)

    def _load_new_model(self):
        """Load a new model for training."""
        model_name = self.config['model']
        print(f"\nLoading model: {model_name}")
        
        # Validate local model files exist (official YOLO models auto-download)
        if not Path(model_name).exists() and not model_name.startswith('yolo'):
            raise FileNotFoundError(f"Model file not found: {model_name}")
        
        return YOLO(model_name)

    # =========================================================================
    # Training Arguments
    # =========================================================================

    def _build_training_args(self, device):
        """Build complete training arguments from config."""
        train_args = self._build_base_args(device)
        self._add_augmentation_args(train_args)
        self._add_hyperparameter_args(train_args)
        return train_args

    def _build_base_args(self, device):
        """Build base training arguments."""
        train_config = self.config.get('training', {})
        
        args = {
            'data': self.config['data_yaml'],
            'epochs': train_config.get('epochs'),
            'imgsz': train_config.get('imgsz'),
            'batch': train_config.get('batch_size'),
            'device': device,
            'workers': train_config.get('workers'),
            'project': str(self.project_dir),
            'name': self.config['experiment_name'],
            'exist_ok': True,
            'save': True,
            'verbose': train_config.get('verbose', True),
            'seed': train_config.get('seed'),
            'patience': train_config.get('patience'),
            'save_period': train_config.get('save_period'),
            'pretrained': train_config.get('pretrained'),
            'optimizer': train_config.get('optimizer'),
            'cache': train_config.get('cache'),
            'amp': train_config.get('amp'),
            'deterministic': train_config.get('deterministic'),
        }
        
        # Remove None values
        return {k: v for k, v in args.items() if v is not None}

    def _add_augmentation_args(self, train_args):
        """Add augmentation arguments to training args."""
        self._add_config_section(train_args, 'augmentation')

    def _add_hyperparameter_args(self, train_args):
        """Add hyperparameter arguments to training args."""
        self._add_config_section(train_args, 'hyperparameters')

    def _add_config_section(self, train_args, section_name):
        """Add all non-None values from a config section to train_args."""
        section = self.config.get(section_name, {})
        for key, value in section.items():
            if value is not None:
                train_args[key] = value

    # =========================================================================
    # Device Management
    # =========================================================================

    def _resolve_device(self):
        """Resolve device from config with automatic fallback."""
        device_config = self.config.get('training', {}).get('device')
        
        if device_config is None or str(device_config).lower() == 'auto':
            return self._auto_select_device()
        
        return self._validate_device(device_config)

    def _auto_select_device(self):
        """Automatically select best available device."""
        if torch.cuda.is_available():
            return 'auto'
        if self._is_mps_available():
            return 'mps'
        return 'cpu'

    def _validate_device(self, device_config):
        """Validate requested device is available."""
        device_str = str(device_config).lower()
        
        # Check CUDA availability
        if device_str not in ['cpu', 'mps'] and not torch.cuda.is_available():
            print(f"Warning: CUDA device '{device_config}' requested but unavailable. Using CPU.")
            return 'cpu'
        
        # Check MPS availability
        if device_str == 'mps' and not self._is_mps_available():
            print("Warning: MPS requested but unavailable. Using CPU.")
            return 'cpu'
        
        return device_config
    
    def _is_mps_available(self):
        """Check if Apple MPS is available."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # =========================================================================
    # Display / Logging
    # =========================================================================

    def _print_header(self, text):
        """Print formatted section header."""
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)

    def _print_config_summary(self, device):
        """Print configuration summary."""
        train_config = self.config.get('training', {})
        
        print("\nConfiguration Summary:")
        print("-" * 60)
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Model: {self.config['model']}")
        print(f"Data: {self.config['data_yaml']}")
        print(f"Epochs: {train_config.get('epochs', 'default')}")
        print(f"Batch Size: {train_config.get('batch_size', 'default')}")
        print(f"Image Size: {train_config.get('imgsz', 'default')}")
        print(f"Device: {device}")
        print(self._get_gpu_info(device))
        print("-" * 60)

    def _get_gpu_info(self, device):
        """Get formatted GPU information string."""
        if torch.cuda.is_available():
            device_id = 0 if device in ['auto', 0, '0'] else device
            if str(device_id).isdigit():
                device_id = int(device_id)
            name = torch.cuda.get_device_name(device_id)
            memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            return f"GPU: {name} ({memory:.2f} GB)"
        
        if self._is_mps_available():
            return "GPU: MPS (Apple Silicon)"
        
        return "GPU: None (CPU only)"

    def _print_training_args(self, train_args):
        """Print training arguments in readable format."""
        print("\nTraining Arguments:")
        print(json.dumps({k: str(v) for k, v in train_args.items()}, indent=2))

"""
Experiment setup and summary utilities for training runs.
"""

from datetime import datetime
from pathlib import Path
import json
import shutil
from typing import Dict, Any, Tuple

import torch


class ExperimentManager:
    """Handles experiment directory setup and summary writing."""

    def __init__(self, config: Dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = Path(config_path)
        self.exp_dir: Path = Path()

    def setup_experiment(self) -> Path:
        """Create experiment directory, copy config, and return path."""
        exp_settings = self.config.get('experiment', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['experiment_name']

        exp_base_dir = Path(exp_settings.get('output_dir', 'experiments'))
        self.exp_dir = exp_base_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = self.exp_dir / 'config.yaml'
        shutil.copy(self.config_path, config_save_path)

        print(f"Experiment directory: {self.exp_dir}")
        print(f"Configuration saved to: {config_save_path}")

        return self.exp_dir

    def save_training_summary(self, train_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
        """Persist training summary metadata to disk."""
        summary = {
            'experiment_name': self.config['experiment_name'],
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'train_args': {k: str(v) for k, v in train_args.items()},
            'system_info': self._system_info(),
        }

        summary_path = self.exp_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining summary saved to: {summary_path}")
        print(
            f"Model weights saved in: {self.exp_dir / 'runs' / 'train' / 'weights'}")

        return summary, summary_path

    @staticmethod
    def _system_info() -> Dict[str, Any]:
        """Collect system info for reproducibility."""
        return {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }

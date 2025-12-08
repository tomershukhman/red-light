"""
Red Light Violation Detection - Training Script

Thin CLI wrapper that delegates training to the package code.
"""

import argparse
import sys

from dotenv import load_dotenv

# Load environment variables early so settings like PYTORCH_CUDA_ALLOC_CONF
# take effect before torch/ultralytics initialize.
load_dotenv()

from red_light.config import ConfigError
from red_light.training import RedLightDetectionTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for red light violation detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Uses configs/train_config.yaml
  python train.py --config configs/my_experiment.yaml  # Uses custom config

All training parameters are specified in the config YAML file.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training configuration YAML file (default: configs/train_config.yaml)'
    )

    args = parser.parse_args()

    try:
        # Create trainer
        trainer = RedLightDetectionTrainer(args.config)

        # Train model
        trainer.train()

        print("\n" + "="*60)
        print("Training pipeline completed successfully!")
        print("="*60)
        print(f"\nExperiment directory: {trainer.exp_dir}")
        print(
            f"Best model: {trainer.exp_dir / 'runs' / 'train' / 'weights' / 'best.pt'}")
        print(
            f"Last model: {trainer.exp_dir / 'runs' / 'train' / 'weights' / 'last.pt'}")

        return 0

    except (ConfigError, Exception) as e:
        print(f"\n{'='*60}")
        print("ERROR: Training failed!")
        print("="*60)
        print(f"{str(e)}")
        print("\nPlease check your configuration file and ensure all required fields are present.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

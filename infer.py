"""
Red Light Violation Detection - Inference Script

Thin CLI wrapper that delegates inference to the package code.
"""

import argparse
import sys

from red_light.config import ConfigError
from red_light.inference import run_inference


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained red light detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --model path/to/best.pt --source image.jpg
  python infer.py --model path/to/best.pt --source images_dir/

All inference parameters are configured in infer_config.yaml
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (best.pt or last.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image or directory of images'
    )
    parser.add_argument(
        '--infer-config',
        type=str,
        default='configs/infer_config.yaml',
        help='Path to inference configuration file (default: configs/infer_config.yaml)'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save detection results as text files'
    )

    args = parser.parse_args()

    try:
        run_inference(
            model_path=args.model,
            source=args.source,
            infer_config_path=args.infer_config,
            save_txt=args.save_txt,
        )
        return 0

    except (ConfigError, FileNotFoundError, Exception) as e:
        print(f"\n{'='*60}")
        print("ERROR: Inference failed!")
        print("="*60)
        print(f"{str(e)}")
        print("\nPlease check your configuration files and paths.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

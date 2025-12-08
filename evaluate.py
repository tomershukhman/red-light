"""
Red Light Violation Detection - Evaluation Script

Thin CLI wrapper that delegates evaluation to the package code.
"""

import argparse
import sys

from red_light.config import ConfigError
from red_light.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained YOLOv8 model for red light detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model path/to/best.pt --split val
  python evaluate.py --model path/to/best.pt --split test

All evaluation parameters are configured in eval_config.yaml
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (best.pt or last.pt)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/data.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--eval-config',
        type=str,
        default='configs/eval_config.yaml',
        help='Path to evaluation configuration file'
    )

    args = parser.parse_args()

    try:
        # Create evaluator
        evaluator = ModelEvaluator(args.model, args.data, args.eval_config)

        # Run evaluation
        evaluator.evaluate(split=args.split)

        print("\n" + "="*60)
        print("Evaluation completed successfully!")
        print("="*60)
        print(f"Results saved to: {evaluator.output_dir}")

        return 0

    except (ConfigError, FileNotFoundError, Exception) as e:
        print(f"\n{'='*60}")
        print("ERROR: Evaluation failed!")
        print("="*60)
        print(f"{str(e)}")
        print("\nPlease check your configuration files and model path.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

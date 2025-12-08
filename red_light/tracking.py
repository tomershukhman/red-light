"""
WandB setup for YOLO training.
See: https://docs.ultralytics.com/integrations/weights-biases/
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def setup_wandb(config: dict) -> None:
    """
    Enable wandb in YOLO settings and set project/entity from config.

    According to docs, YOLO uses 'project' and 'name' train args for wandb.
    We set WANDB_PROJECT env var to override with tracking.project from config.
    """
    # Load .env for WANDB_API_KEY
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    tracking = config.get('tracking', {})
    if not tracking.get('enabled'):
        return

    # Enable wandb in YOLO settings
    from ultralytics import settings
    settings.update({'wandb': True})

    # Set wandb project from config (overrides train's 'project' arg for wandb)
    if project := tracking.get('project'):
        os.environ['WANDB_PROJECT'] = project
    if entity := tracking.get('entity'):
        os.environ['WANDB_ENTITY'] = entity

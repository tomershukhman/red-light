"""
WandB setup for YOLO training.
See: https://docs.ultralytics.com/integrations/weights-biases/
"""

from pathlib import Path
from dotenv import load_dotenv


def setup_wandb(config: dict) -> None:
    """
    Initialize wandb with correct project/entity before YOLO training.

    YOLO's wandb callback ignores WANDB_PROJECT env var and uses the 'project'
    argument from train(). To override this, we call wandb.init() directly.
    YOLO's callback will detect the existing run and use it.
    """
    # Load .env for WANDB_API_KEY
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    tracking = config.get('tracking', {})
    if not tracking.get('enabled'):
        return

    # Enable wandb in YOLO settings
    from ultralytics import settings
    settings.update({'wandb': True})

    # Initialize wandb directly - YOLO will use this existing run
    import wandb
    
    wandb.init(
        project=tracking.get('project'),
        entity=tracking.get('entity'),
        name=config.get('experiment_name'),
        tags=tracking.get('tags'),
        notes=tracking.get('notes'),
        mode=tracking.get('mode', 'online'),
    )

"""
Experiment tracking backends.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class WandbTracker:
    """Thin wrapper around Weights & Biases logging."""

    def __init__(self, tracking_config: Dict[str, Any], exp_dir: Path):
        # Load environment variables (e.g., WANDB_API_KEY) from project-level .env
        project_root = Path(__file__).resolve().parents[1]
        env_path = project_root / ".env"
        load_dotenv(env_path)

        self.config = tracking_config or {}
        self.exp_dir = Path(exp_dir)
        self.enabled = bool(self.config.get('enabled', False))
        self.wandb_run = None
        self.wandb = None

        if not self.enabled:
            print("Experiment tracking: disabled (set tracking.enabled: true to enable).")
            return

        provider = self.config.get('provider', 'wandb')
        if provider != 'wandb':
            print(
                f"Experiment tracking provider '{provider}' not supported. "
                "Supported providers: wandb. Tracking disabled for this run."
            )
            self.enabled = False
            return

        try:
            import wandb  # type: ignore

            self.wandb = wandb
            print("Experiment tracking: enabled with Weights & Biases.")
        except ImportError:
            print(
                "Experiment tracking requested but `wandb` is not installed. "
                "Install with `pip install wandb` or disable tracking."
            )
            self.enabled = False

    def start(self, resolved_device: Any, train_args: Dict[str, Any], full_config: Dict[str, Any]) -> Optional[Any]:
        """Start a W&B run if enabled."""
        if not self.enabled or not self.wandb:
            return None

        tracking_cfg = self.config
        run_name = tracking_cfg.get('run_name') or self.exp_dir.name
        run_name_suffix = tracking_cfg.get('run_name_suffix')
        if run_name_suffix:
            run_name = f"{run_name}-{run_name_suffix}"

        project = tracking_cfg.get('project', 'red-light-violation')
        entity = tracking_cfg.get('entity')
        tags = tracking_cfg.get('tags') or []
        notes = tracking_cfg.get('notes')
        mode = tracking_cfg.get('mode', 'online')

        print(
            f"Tracking enabled: sending run to W&B "
            f"(project={project}, entity={entity}, name={run_name}, mode={mode})"
        )

        try:
            self.wandb_run = self.wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config={
                    'experiment': full_config,
                    'train_args': train_args,
                    'resolved_device': str(resolved_device),
                },
                tags=tags,
                notes=notes,
                mode=mode,
                dir=str(self.exp_dir),
                reinit=True,
            )

            return self.wandb_run
        except Exception as exc:
            print(
                f"Warning: Failed to initialize Weights & Biases tracking ({exc}). "
                "Continuing without tracking."
            )
            self.enabled = False
            self.wandb_run = None
            return None

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Update run summary if tracking is enabled."""
        if not self.enabled or not self.wandb_run or not self.wandb:
            return
        try:
            self.wandb_run.summary.update(summary)
        except Exception as exc:
            print(f"Warning: Failed to update W&B summary: {exc}")

    def log_artifacts(self, summary_path: Optional[Path]) -> None:
        """Upload config, summary, and weights as artifacts."""
        if not self.enabled or not self.wandb_run or not self.wandb:
            return

        if not self.config.get('log_artifacts', True):
            return

        try:
            artifact = self.wandb.Artifact(
                name=self.exp_dir.name,
                type='model',
                metadata={'experiment_dir': str(self.exp_dir)}
            )

            config_path = self.exp_dir / 'config.yaml'
            if config_path.exists():
                artifact.add_file(str(config_path), name='config.yaml')

            if summary_path and Path(summary_path).exists():
                artifact.add_file(str(summary_path),
                                  name='training_summary.json')

            weights_dir = self.exp_dir / 'runs' / 'train' / 'weights'
            for weight_name in ['best.pt', 'last.pt']:
                weight_path = weights_dir / weight_name
                if weight_path.exists():
                    artifact.add_file(str(weight_path),
                                      name=f"weights/{weight_name}")

            self.wandb_run.log_artifact(artifact)
        except Exception as exc:
            print(f"Warning: Failed to log artifacts to W&B: {exc}")

    def finish(self) -> None:
        """Cleanly finish the tracking run."""
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            finally:
                self.wandb_run = None

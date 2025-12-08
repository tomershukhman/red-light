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
        # Path(__file__).resolve().parents[1] goes from red_light/tracking.py -> red-light/
        project_root = Path(__file__).resolve().parents[1]
        env_path = project_root / ".env"
        # Only try to load if .env exists (wandb can also use system env vars)
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Silently use system environment variables if .env doesn't exist
            load_dotenv()  # This will search in parent directories

        self.config = tracking_config or {}
        self.exp_dir = Path(exp_dir)
        self.enabled = bool(self.config.get('enabled', False))
        self.wandb = None
        self._stored_config = {}  # Store config for later use

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
            
            # Auto-enable wandb in YOLO settings if not already enabled
            self._ensure_yolo_wandb_enabled()
            
            print("Experiment tracking: enabled with Weights & Biases.")
        except ImportError:
            print(
                "Experiment tracking requested but `wandb` is not installed. "
                "Install with `pip install wandb` or disable tracking."
            )
            self.enabled = False

    def _ensure_yolo_wandb_enabled(self) -> None:
        """
        Automatically enable WandB in YOLO settings if not already enabled.
        This ensures wandb logging works without manual intervention.
        """
        try:
            from ultralytics.utils import SettingsManager
            
            settings = SettingsManager()
            if not settings.get('wandb', False):
                print("  → Enabling WandB in YOLO settings automatically...")
                settings['wandb'] = True
                settings.save()
                print("  ✓ WandB enabled in YOLO settings")
            else:
                print("  ✓ WandB already enabled in YOLO settings")
        except Exception as exc:
            print(f"  ⚠ Could not auto-enable WandB in YOLO settings: {exc}")
            print("  → Run manually: yolo settings wandb=True")
    
    def get_yolo_wandb_config(self) -> Optional[Dict[str, str]]:
        """
        Set environment variables for YOLO's WandB integration.
        
        YOLO reads WandB config from environment variables when wandb=True.
        """
        if not self.enabled or not self.wandb:
            return None
            
        import os
        
        tracking_cfg = self.config
        run_name = tracking_cfg.get('run_name') or self.exp_dir.name
        run_name_suffix = tracking_cfg.get('run_name_suffix')
        if run_name_suffix:
            run_name = f"{run_name}-{run_name_suffix}"

        wandb_project = tracking_cfg.get('project', 'red-light-violation')
        wandb_entity = tracking_cfg.get('entity')
        
        # Set environment variables that YOLO's WandB integration will read
        os.environ['WANDB_PROJECT'] = wandb_project
        if wandb_entity:
            os.environ['WANDB_ENTITY'] = wandb_entity
        os.environ['WANDB_NAME'] = run_name
        
        print(
            f"✓ WandB environment configured "
            f"(project={wandb_project}, entity={wandb_entity}, run={run_name})"
        )
        print("  YOLO will automatically log all training metrics to WandB")
        
        return {}  # No train_args needed, using env vars
    
    def start(self, resolved_device: Any, train_args: Dict[str, Any], full_config: Dict[str, Any]) -> Optional[Any]:
        """
        Store configuration for later use (YOLO handles wandb initialization).
        """
        if not self.enabled or not self.wandb:
            return None
            
        # Store config for later summary logging
        self._stored_config = {
            'experiment': full_config,
            'train_args': train_args,
            'resolved_device': str(resolved_device),
        }
        
        return True

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Update run summary if tracking is enabled."""
        if not self.enabled or not self.wandb:
            return
        try:
            # YOLO's wandb run should be active at this point
            if self.wandb.run is not None:
                self.wandb.run.summary.update(summary)
                print("✓ Training summary logged to W&B")
            else:
                print("⚠ No active W&B run to update summary")
        except Exception as exc:
            print(f"Warning: Failed to update W&B summary: {exc}")

    def log_artifacts(self, summary_path: Optional[Path]) -> None:
        """Upload config, summary, and weights as artifacts."""
        if not self.enabled or not self.wandb:
            return

        if not self.config.get('log_artifacts', True):
            return

        try:
            # YOLO's wandb run should be active at this point
            if self.wandb.run is None:
                print("⚠ No active W&B run to log artifacts")
                return
                
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

            self.wandb.run.log_artifact(artifact)
            print("✓ Artifacts logged to W&B")
        except Exception as exc:
            print(f"Warning: Failed to log artifacts to W&B: {exc}")

    def finish(self) -> None:
        """Cleanly finish the tracking run (YOLO handles this automatically)."""
        if not self.enabled or not self.wandb:
            return
            
        # YOLO automatically finishes its wandb run, but we can explicitly
        # close it here if needed
        if self.wandb.run is not None:
            try:
                print("✓ W&B run finished (managed by YOLO)")
                # Note: YOLO typically handles finish() itself
            except Exception as exc:
                print(f"Warning: Issue with W&B finish: {exc}")

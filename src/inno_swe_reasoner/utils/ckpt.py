from pathlib import Path
import torch
import torch.nn as nn
import json
import shutil

from inno_swe_reasoner.config import WeightCheckpointConfig
from inno_swe_reasoner.utils.logger import get_logger
from inno_swe_reasoner.utils.utils import get_ckpt_dir


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, output_dir: Path, config: WeightCheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self._logger = get_logger()
        self.ckpt_steps: list[int] = []

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def save(self, model: nn.Module, step: int, optimizer=None, **extra_metadata):
        """Save model checkpoint compatible with HuggingFace."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Saving checkpoint at step {step} to {ckpt_path}")

        # Save model state dict (PyTorch format)
        model_path = ckpt_path / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)

        # If the model has a save_pretrained method (HuggingFace models), use it
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt_path)
            self._logger.info(f"Saved HuggingFace model to {ckpt_path}")

        # Save optimizer state if provided
        if optimizer is not None:
            optimizer_path = ckpt_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)

        # Save training metadata
        metadata = {
            "step": step,
            **extra_metadata,
        }
        metadata_path = ckpt_path / "training_state.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Track saved checkpoints
        self.ckpt_steps.append(step)

        # Clean up old checkpoints if needed
        if self.config.keep_last_n is not None:
            self._cleanup_old_checkpoints()

        self._logger.success(f"Checkpoint saved at step {step}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.ckpt_steps) > self.config.keep_last_n:
            # Sort to get oldest first
            sorted_steps = sorted(self.ckpt_steps)
            steps_to_remove = sorted_steps[: -self.config.keep_last_n]

            for step in steps_to_remove:
                ckpt_path = self.get_ckpt_path(step).parent  # Get step_X directory
                if ckpt_path.exists():
                    shutil.rmtree(ckpt_path)
                    self._logger.info(f"Removed old checkpoint at step {step}")
                self.ckpt_steps.remove(step)

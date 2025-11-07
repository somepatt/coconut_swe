from pathlib import Path


def get_ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"

from pathlib import Path

from pydantic import Field
from typing import Annotated
from inno_swe_reasoner.utils.pydantic_config import BaseConfig, BaseSettings
from inno_swe_reasoner.config import ModelConfig, AdamWConfig, OptimizerConfigType
from inno_swe_reasoner.config import WandbMonitorConfig, WeightCheckpointConfig


class CoconutDataConfig(BaseConfig):
    """Configuration class for COCONUT SFT data processing."""

    # Name of the dataset on huggingface
    name: str = "SWE-Swiss/SWESwiss-SFT-Repair-4K"
    # Split to use
    split: str = "train"
    # Maximum sequence length
    max_seq_length: int = 512
    # Number of samples to use (for debugging, None means all)
    num_samples: int | None = None
    # if to shuffle the dataset
    shuffle: bool = True
    # batch size
    batch_size: int = 1
    # if to use mock data
    mock_data: bool = False
    # seed
    seed: int = 42
    # max epochs
    max_epochs: int = 3
    # COCONUT specific parameters
    # num coconut stages
    num_stages: int = 3
    # number of continuous thoughts per step
    c: int = 3
    epoch_per_stage: int = 2
    gradient_accumulation_steps: int = 20
    quantization: str | None = "4bit"


class CoconutTrainerConfig(BaseSettings):
    """Configuration class for COCONUT trainer."""

    # model related config
    model: ModelConfig = ModelConfig()

    # data related config
    data: CoconutDataConfig = CoconutDataConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The wandb configuration
    wandb: WandbMonitorConfig | None = None

    # The weight checkpoint configuration
    checkpoint: WeightCheckpointConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

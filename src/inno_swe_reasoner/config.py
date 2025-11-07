# Adapted from from https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/config.py

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from inno_swe_reasoner.utils.pydantic_config import BaseConfig

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]

MOE_MODEL_MAPS = {
    "Qwen/Qwen3-30B-A3B": "Jackmin108/Qwen3-30B-A3B-Fast",
    "moonshotai/Moonlight-16B-A3B-Instruct": "Jackmin108/Moonlight-16B-A3B-Instruct-Fast",
}


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    attn: Annotated[
        AttnImplementation, Field(description="The attention implementation to use.")
    ] = "flash_attention_2"

    quantization:  Annotated[
        Literal["4bit", "8bit", None],
        Field(description="The quantization.")
    ] = "4bit"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for model and tokenizer initialization.",
        ),
    ] = False

    impl: Annotated[
        Literal["hf", "liger_kernel", "custom"],
        Field(
            description="Whether to use Liger Kernel.",
        ),
    ] = "hf"

    load_using_meta: Annotated[
        bool,
        Field(
            description="Whether to load the model using meta device then load from HF ckpt.",
        ),
    ] = False

    optimization_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model optimization.",
        ),
    ] = "bfloat16"

    reduce_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model reduce.",
        ),
    ] = "bfloat16"

    @model_validator(mode="after")
    def _map_model_name_for_moe(self):
        """Map model name if it exists in MOE_MODEL_MAPS."""
        if self.name in MOE_MODEL_MAPS:
            self.name = MOE_MODEL_MAPS[self.name]
        return self

    @model_validator(mode="after")
    def trust_remote_code_only_with_hf(self):
        """Trust remote code only if the model is from HF."""
        if self.trust_remote_code:
            if self.impl != "hf":
                raise ValueError(
                    "Trust remote code is only supported with the HF implementation."
                )
        return self


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[
        int,
        Field(
            ge=0, description="Number of warmup steps for the learning rate scheduler."
        ),
    ] = 10

    decay_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Number of steps to decay the learning rate during the final portion of training.",
        ),
    ] = 10

    min_lr: Annotated[
        float, Field(ge=0, description="Minimum learning rate to converge to.")
    ] = 0.0


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[
        int,
        Field(
            ge=0, description="Number of warmup steps for the learning rate scheduler."
        ),
    ] = 10

    min_lr: Annotated[
        float, Field(ge=0, description="Minimum learning rate to converge to.")
    ] = 0.0


SchedulerConfigType: TypeAlias = (
    ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig
)


class BaseOptimizerConfig(BaseModel):
    lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    max_norm: Annotated[
        float, Field(ge=0, description="Maximum gradient norm to clip.")
    ] = 1.0


class SGDConfig(BaseOptimizerConfig):
    type: Literal["sgd"] = "sgd"
    nesterov: bool = True
    momentum: float = 0.9


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


OptimizerConfigType: TypeAlias = SGDConfig | AdamWConfig | MuonConfig


class LogExtrasConfig(BaseConfig):
    """Configures extra logging for W&B tables."""

    samples: Annotated[
        bool,
        Field(
            description="Whether to log prompt/response samples to W&B tables.",
        ),
    ] = True

    distributions: Annotated[
        bool,
        Field(
            description="Whether to log distributions (like rewards, advantages, etc.) to W&B tables.",
        ),
    ] = True

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log extras to W&B table.",
        ),
    ] = 10


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: Annotated[str, Field(description="The W&B project to log to.")] = (
        "prime-rl"
    )

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    offline: Annotated[
        bool, Field(description="Whether to run W&B in offline mode.")
    ] = False

    # Individual configs (can only be specified on trainer or orchestrator)
    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras to W&B tables. If None, no extras are logged.",
        ),
    ] = LogExtrasConfig()


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the training checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

    keep_last_n: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of previous checkpoints to keep",
        ),
    ] = None

from torch import nn
from torch.optim import SGD, AdamW, Optimizer

from inno_swe_reasoner.config import OptimizerConfigType


def setup_optimizer(config: OptimizerConfigType, model: nn.Module) -> Optimizer:
    match config.type:
        case "sgd":
            return SGD(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )

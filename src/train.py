import time
import torch

from torch.nn.functional import cross_entropy
from inno_swe_reasoner.coconut.config import CoconutTrainerConfig
from inno_swe_reasoner.utils.pydantic_config import parse_argv
from inno_swe_reasoner.coconut.model import setup_model, setup_tokenizer
from inno_swe_reasoner.utils.logger import setup_logger
from inno_swe_reasoner.optim import setup_optimizer
from inno_swe_reasoner.coconut.coconut_utils import calculate_eot_offset
from inno_swe_reasoner.utils.monitor import setup_monitor
from inno_swe_reasoner.utils.ckpt import CheckpointManager
from inno_swe_reasoner.coconut.data import (
    setup_dataset,
    setup_dataloader,
    tokenize_data,
)


def train(config: CoconutTrainerConfig):
    # Setup logger
    logger = setup_logger(
        log_level="INFO"
    )  # TODO: Make log file and log level configurable
    # Setup model
    logger.info("Setting up model...")
    model = setup_model(config.model)

    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = setup_tokenizer(config.model)

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(
        config.wandb, output_dir=config.output_dir, run_config=config
    )

    # Setup the checkpoint manager
    weightckpt_manager = CheckpointManager(
        output_dir=config.output_dir, config=config.checkpoint
    )

    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model)

    # Setup dataset and dataloader
    logger.info("Setting up dataset...")
    dataset = setup_dataset(config.data, tokenizer)
    dataloader = setup_dataloader(dataset, config.data)

    eot_offset = calculate_eot_offset(tokenizer)

    # COCONUT Training
    step = 0
    for stage in range(1, config.data.num_stages + 1):
        if stage > 0:
            logger.info(f"Resetting optimizer at Stage {stage}")
            optimizer = setup_optimizer(config.optim, model)

        for epochs in range(config.data.epoch_per_stage):
            logger.info(f"Starting epoch {epochs} at stage {stage}")
            for batch in dataloader:
                mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")
                step += 1
                step_start_time = time.time()
                batch_loss = 0.0

                # This goes through each datapoint one by one
                # Justification is that CoT on SWE_SWISS_SFT dataset is easily > 250K.
                # Might revisit this decision though.
                for idx, (prompt, cot_steps, answer) in enumerate(
                    zip(batch["prompt"], batch["cot_steps"], batch["answer"])
                ):
                    # Determine how many steps to replace with continuous thoughts
                    steps_to_replace = min(stage, len(cot_steps))
                    num_continuous_thoughts = steps_to_replace * config.data.c
                    remaining_cot = cot_steps[steps_to_replace:]

                    if stage == 0:
                        # Stage 0: Regular CoT training (no continuous thoughts)
                        input_ids, loss_mask = tokenize_data(
                            tokenizer, prompt, cot_steps, answer, max_length=config.data.max_seq_length
                        )

                        torch.cuda.empty_cache()
                        
                        target_ids = torch.tensor(input_ids.copy()[1:]).unsqueeze(0).to("cuda")
                        input_ids = torch.tensor(input_ids[:-1]).unsqueeze(0).to("cuda")
                        loss_mask = torch.tensor(loss_mask[:-1]).unsqueeze(0).to("cuda")

                        position_ids = (
                            torch.tensor(list(range(input_ids.shape[1])))
                            .unsqueeze(0)
                            .to("cuda")
                        )

                        assert input_ids.shape == target_ids.shape == loss_mask.shape, (
                            f"input_ids, loss_mask and target_ids must have the same length, but got {input_ids.shape=}, {loss_mask.shape=}, {target_ids.shape=}"
                        )

                        # run forward pass on the tokens
                        logits = model.embed_and_forward(input_ids, position_ids, False)
                        B, L, V = logits.shape
                        loss = cross_entropy(
                            logits.view(-1, V), target_ids.view(-1), reduction="none"
                        ).view(B, L)

                        del logits
                        del input_ids
                        del position_ids
                        del target_ids
                        torch.cuda.empty_cache()

                        loss = loss[loss_mask].mean()
                        batch_loss += loss
                        logger.debug(f"Example {idx} LOSS = {loss.item()}")
                    else:
                        input_ids, _ = tokenize_data(
                            tokenizer=tokenizer, prompt=prompt + "<bot>", max_length=config.data.max_seq_length
                        )
                        input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")
                        input_embed = model.get_embeddings(input_ids)

                        torch.cuda.empty_cache()

                        # Generate continuous thoughts
                        num_continuous_thoughts = 2  # TODO: Remove this later.
                        for i in range(num_continuous_thoughts):
                            hidden_states = model.forward_from_embeddings(
                                input_embed, None, True
                            )
                            last_hidden = hidden_states[:, -1, :].unsqueeze(-2)
                            input_embed = torch.cat([input_embed, last_hidden], dim=-2)

                        eot_cot_toks, loss_mask = tokenize_data(
                            tokenizer=tokenizer,
                            cot_steps=["<eot>"] + remaining_cot,
                            answer=answer,
                        )
                        eot_cot_toks = (
                            torch.tensor(eot_cot_toks[:-1]).unsqueeze(0).to("cuda")
                        )
                        target_eot_cot = eot_cot_toks[:, 1:]
                        eot_cot_toks = eot_cot_toks[:, :-1]

                        assert target_eot_cot.shape == eot_cot_toks.shape, (
                            "Tensor for input and target tokens for post latent thoughts should have the same shape"
                        )

                        eot_cot_embed = model.get_embeddings(eot_cot_toks)

                        post_latent_embed = torch.cat(
                            [input_embed, eot_cot_embed], dim=-2
                        )

                        logits = model.forward_from_embeddings(
                            post_latent_embed, None, False
                        )
                        B, L, V = logits.shape

                        num_prefix_tokens = input_embed.shape[-2]
                        dummy_targets = torch.zeros(
                            (B, num_prefix_tokens), dtype=torch.long, device="cuda"
                        )
                        target_ids = torch.cat([dummy_targets, target_eot_cot], dim=-1)
                        num_prefix_tokens += eot_offset

                        # Build loss mask: False for everything before <eot>, True for <eot> onward
                        loss_mask = (
                            torch.cat(
                                [
                                    torch.zeros(
                                        num_prefix_tokens - 1, dtype=torch.bool
                                    ),
                                    torch.tensor(
                                        loss_mask[eot_offset:], dtype=torch.bool
                                    ),
                                ]
                            )
                            .unsqueeze(0)
                            .to("cuda")
                        )

                        loss = cross_entropy(
                            logits.view(-1, V), target_ids.view(-1), reduction="none"
                        ).view(B, L)

                        del logits
                        del input_ids
                        del target_ids
                        torch.cuda.empty_cache()

                        loss = loss[loss_mask].mean()
                        batch_loss += loss

                # After processing all examples in batch:
                batch_loss = batch_loss / config.data.gradient_accumulation_steps
                batch_loss.backward()
                
                batch_loss_value = float(batch_loss.detach().item())
                del batch_loss
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.optim.max_norm
                )
                if step % config.data.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                step_time = time.time() - step_start_time

                log_metrics = {
                    "loss/batch_mean": batch_loss_value,
                    "optim/grad_norm": grad_norm.item(),
                    "optim/learning_rate": optimizer.param_groups[0]["lr"],
                    "timing/step_time": step_time,
                    "timing/examples_per_sec": len(batch["prompt"]) / step_time,
                    "training/stage": stage,
                    "training/batch_size": len(batch["prompt"]),
                }

                monitor.log(log_metrics)
                step_message = f"Step {step} | Time: {step_time:.2f}s | Loss: {batch_loss_value:.4f} | Grad. Norm: {grad_norm:.4f}"
                logger.success(step_message)

        weightckpt_manager.save(model=model, step=step)


def main():
    train(parse_argv(CoconutTrainerConfig))


if __name__ == "__main__":
    main()

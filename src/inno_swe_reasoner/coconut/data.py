# Adapted from https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/sft/data.py
import re
import json
from typing import Dict, List, Optional
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset
from datasets import Dataset, load_dataset

from inno_swe_reasoner.coconut.config import CoconutDataConfig
from inno_swe_reasoner.utils.logger import get_logger


class SFTDataset(IterableDataset):
    """Dataset wrapping HF SFT dataset with `message` format."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        shuffle: bool = True,
        max_epochs: int | None = None,
        seed: int = 42,
        seq_len: int = 2048,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.num_examples = len(dataset)
        self.step = 0
        self.epoch = 0
        self.max_epochs = max_epochs
        self.seq_len = seq_len
        self.logger = get_logger()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def debug_example(self, idx=0):
        """Debug helper to check example structure"""
        example = self.dataset[idx]
        print(f"Example type: {type(example)}")
        print(
            f"Example keys: {example.keys() if isinstance(example, dict) else 'Not a dict'}"
        )
        print(f"Example type of value = {type(example['messages'])}")
        return example

    def _extract_structured_components(self, messages: list[dict]) -> dict:
        prompt = next(msg["content"] for msg in messages if msg["role"] == "user")
        assistant_msg = next(
            msg["content"] for msg in messages if msg["role"] == "assistant"
        )

        # Parse think tags
        think_match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)

        if think_match:
            think_content = think_match.group(1).strip()
            answer = assistant_msg.replace(think_match.group(0), "").strip()
            cot_steps = [
                step.strip() for step in think_content.split("\n") if step.strip()
            ]
        else:
            cot_steps = []
            answer = assistant_msg.strip()

        return {
            "prompt": prompt,
            "cot_steps": cot_steps,
            "answer": answer,
            "messages": messages,  # Keep original for flexible tokenization
        }

    def _process(self, example: dict):
        # Assumes that the key 'messages' exists in the example dict
        if "messages" not in example:
            raise ValueError(
                "All examples must have a 'messages' column for SFT training."
            )

        # messages is string-ified list of dicts, parse it
        def parse_messages(example):
            example["messages"] = json.loads(example["messages"])
            return example

        def strip_content(messages: list[dict]) -> list[dict]:
            def _strip_content(message: dict) -> dict:
                if isinstance(message.get("content"), str):
                    return {**message, "content": message["content"].strip()}
                return message

            return [_strip_content(message) for message in messages]

        example = parse_messages(example)
        messages = strip_content(example["messages"])

        structured_example = self._extract_structured_components(messages)
        return structured_example

    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = (
            self.dataset.shuffle(seed=self.epoch + self.seed)
            if self.shuffle
            else self.dataset
        )
        for idx in range(self.num_examples):
            self.step += 1
            example = dataset[idx]

            processed_example = self._process(example)

            if processed_example is None:
                continue

            yield processed_example
        # while True:
        #     self.step += 1

        #     # Determine epoch from current step
        #     epoch = (self.step - 1) // self.num_examples

        #     # Break if max epochs reached
        #     if self.max_epochs is not None and epoch >= self.max_epochs:
        #         break

        #     # Update stored epoch if new epoch is reached, optionally shuffle
        #     if epoch > self.epoch:
        #         self.epoch = epoch
        #         dataset = (
        #             self.dataset.shuffle(seed=self.epoch + self.seed)
        #             if self.shuffle
        #             else self.dataset
        #         )

        #     example = dataset[(self.step - 1) % self.num_examples]

        #     processed_example = self._process(example)

        #     if processed_example is None:
        #         continue

        #     yield processed_example


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """
    Simple collation for structured data - just group fields into lists.
    No padding needed since we're not dealing with tensors yet.
    """
    return {
        "prompt": [item["prompt"] for item in batch],
        "cot_steps": [item["cot_steps"] for item in batch],  # List of List[str]
        "answer": [item["answer"] for item in batch],
        "messages": [item["messages"] for item in batch],  # Keep if needed
    }


def setup_dataloader(
    dataset: SFTDataset,
    config: CoconutDataConfig,
) -> DataLoader:
    """
    Create a DataLoader for structured (non-tokenized) data.
    """
    batch_size = getattr(config, "batch_size", 1)
    num_workers = 0  # Keep 0 for iterable datasets
    pin_memory = False  # No tensors to pin

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,  # Much simpler!
        pin_memory=pin_memory,
    )

    return dataloader


def setup_dataset(config: CoconutDataConfig, tokenizer: AutoTokenizer) -> Dataset:
    logger = get_logger()

    if not config.mock_data:
        logger.info(f"Loading dataset from {config.name} split {config.split}...")
        dataset = load_dataset(config.name, split=config.split)
    else:
        logger.info("Loading mock dataset here")
        with open("src/inno_swe_reasoner/coconut/mock_data.json", "r") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    if config.num_samples is not None:
        dataset = dataset.take(config.num_samples)
        logger.info(f"Selected first {config.num_samples} samples from the dataset.")

    return SFTDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=config.max_seq_length,
        seed=config.seed,
        shuffle=config.shuffle,
    )


def tokenize_data(
    tokenizer: AutoTokenizer,
    prompt: str = None,
    cot_steps: Optional[List[str]] = None,
    answer: Optional[str] = None,
    max_length: int = 2048
) -> List[int]:
    """Tokenize a list of prompts."""
    # Build assistant content
    assistant_parts = []
    if cot_steps:
        assistant_parts.append("\n".join(cot_steps))
    if answer:
        assistant_parts.append(answer)

    assistant_content = "\n".join(assistant_parts) if assistant_parts else ""

    # Create messages
    messages = []
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    # Tokenize
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=False,
    )

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    if prompt is not None:
        user_messages = [{"role": "user", "content": prompt}]
        user_ids = tokenizer.apply_chat_template(
            conversation=user_messages,
            add_generation_prompt=True,  # Includes the assistant header
        )
    else:
        user_ids = []

    # Create mask: 0 for user tokens, 1 for assistant tokens
    loss_mask = [False] * len(user_ids) + [True] * (len(input_ids) - len(user_ids))

    return input_ids, loss_mask[1:]

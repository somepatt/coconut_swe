from transformers import AutoTokenizer


def calculate_eot_offset(tokenizer: AutoTokenizer) -> int:
    """Calculate how many tokens to skip for the eot token"""
    return len(tokenizer.encode("<eot>"))

from typing import Any, Dict
from transformers import PreTrainedTokenizerBase

def tokenize_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
) -> Dict[str, Any]:
    # tokenize the "text" field, drop overflowing tokens
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
        "special_tokens_mask": outputs["special_tokens_mask"],
    }

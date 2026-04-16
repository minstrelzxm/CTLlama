from typing import Any, Dict
from transformers import PreTrainedTokenizerBase

def tokenize_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
) -> Dict[str, Any]:
    # Tokenize the "text" field, drop overflowing tokens.
    # No padding here — the collator pads per-batch to the longest sample.
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }

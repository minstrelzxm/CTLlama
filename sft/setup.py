import wandb
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from utils import formatting_prompts_func, tokenize_function

class Setup:
    def __init__(
        self,
        model_name: str,
        token: str,
        max_seq_length: int = 8192 * 2,
        dtype=None,
        load_in_4bit: bool = False,
        peft_config: dict = None,
        data_path: str = "data/sft_data.json"
    ):
        self.model_name = model_name
        self.token = token
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.peft_config = peft_config or {}
        self.data_path = data_path

    def login(self):
        wandb.login()

    def init_model_and_tokenizer(self):
        # Load base model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=self.token
        )
        # Tokenizer setup
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            # LLaMA-3 ships a dedicated right-pad token; fall back to eos if missing.
            pad_token = "<|finetune_right_pad_id|>"
            if pad_token in tokenizer.get_vocab():
                tokenizer.pad_token = pad_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            "additional_special_tokens": ['<begin_of_ref>', '<end_of_ref>','<think>','</think>', '<answer>', '</answer>']
        })
        # (Optional) set up chat template if needed
        model.resize_token_embeddings(len(tokenizer))

        # Apply PEFT
        model = FastLanguageModel.get_peft_model(
            model,
            **self.peft_config
        )
        return model, tokenizer

    def prepare_data(self, tokenizer):
        # Load and format
        dataset = load_dataset("json", data_files=self.data_path, split="train")
        dataset = dataset.map(
            lambda ex: formatting_prompts_func(ex, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )
        # Tokenize (no padding — collator handles it per-batch)
        tokenized = dataset.map(
            lambda ex: tokenize_function(ex, tokenizer, self.max_seq_length),
            batched=True,
            remove_columns=["text"],
        )
        return tokenized

    def get_training_args(self, output_dir: str = "output", **kwargs):
        return TrainingArguments(
            output_dir=output_dir,
            **kwargs
        )

    def get_data_collator(self, tokenizer):
        # Response-only loss: labels are -100 for everything up to and
        # including the assistant-header, so only the response tokens
        # contribute to the SFT loss.
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
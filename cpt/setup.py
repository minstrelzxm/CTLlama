import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from utils import tokenize_function

class SetupCPT:
    def __init__(
        self,
        model_name: str,
        data_file: str = "data/data.json",
        max_seq_length: int = 4096,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: list[str] = None,
    ):
        """
        model_name: HF path or local dir for base LLaMA
        data_file: path to your JSON containing {"text": ...} entries
        max_seq_length: maximum context length (4096)
        LoRA args: r, alpha, and modules to adapt
        """
        self.model_name = model_name
        self.data_file = data_file
        self.max_seq_length = max_seq_length

        # default target modules if not provided
        if lora_target_modules is None:
            lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],

        # load base model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # apply LoRA adapters for CPT
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)

    def prepare_dataset(self, split: str = "train"):
        # load JSON; expects a top-level list or newline-delimited JSON
        ds = load_dataset(
            "json",
            data_files={split: self.data_file},
            split=split
        )
        # tokenize in a single pass
        tokenized = ds.map(
            lambda ex: tokenize_function(ex, self.tokenizer, self.max_seq_length),
            batched=True,
            remove_columns=ds.column_names
        )
        return tokenized

    def save_tokenizer(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.tokenizer.save_pretrained(out_dir)

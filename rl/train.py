from trl import GRPOTrainer
from transformers import PreTrainedTokenizer
from datasets import Dataset
import torch

class RLTrainer:
    def __init__(self, model, tokenizer: PreTrainedTokenizer,
                 train_dataset: Dataset, reward_funcs: list, config):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=train_dataset
        )

    def train(self):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU = {props.name}, Total memory = {round(props.total_memory/(1024**3),2)} GB")
        stats = self.trainer.train()
        return stats

    def save(self, output_dir: str = "grpo_saved_lora"):
        self.model.save_lora(output_dir)
        self.tokenizer.save_pretrained(output_dir)
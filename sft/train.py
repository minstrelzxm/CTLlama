import torch
from transformers import Trainer

class ModelTrainer:
    def __init__(self, model, tokenizer, train_dataset, data_collator, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args
        )

    def train(self):
        # Show initial GPU stats
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU = {gpu_props.name}, Total Memory = {round(gpu_props.total_memory / (1024**3), 2)} GB")

        stats = self.trainer.train()

        # Report training stats
        runtime = stats.metrics.get('train_runtime')
        memory = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
        print(f"Training completed in {runtime} seconds. Peak GPU memory usage: {memory} GB")
        return stats

    def save(self, output_dir: str = "lora_model"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
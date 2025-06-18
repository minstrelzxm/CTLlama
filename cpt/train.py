from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

class CPTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        output_dir: str = "cpt_output",
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 3,
        mlm_probability: float = 0.15,
    ):
        """
        model, tokenizer: from SetupCPT (with LoRA)
        train_dataset: tokenized Dataset
        """
        self.model = model
        self.tokenizer = tokenizer

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

        self.args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )

    def train(self):
        self.trainer.train()

    def save_model(self, out_dir: str):
        # Save both base+LoRA adapter weights
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

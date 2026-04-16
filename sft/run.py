import os

from setup import Setup
from train import ModelTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


def main():
    # Setup configuration
    peft_cfg = {
        'r': 32,
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens","lm_head"
        ],
        'lora_alpha': 32,
        'lora_dropout': 0,
        'bias': "none",
        'use_gradient_checkpointing': "unsloth",
        'random_state': 3407,
        'use_rslora': True,
        'loftq_config': None
    }
    setup = Setup(
        model_name="minstrelzxm/ctllama-8b-base",
        token=os.environ["HF_TOKEN"],
        peft_config=peft_cfg
    )
    setup.login()
    model, tokenizer = setup.init_model_and_tokenizer()
    dataset = setup.prepare_data(tokenizer)
    training_args = setup.get_training_args(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=7e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_steps=200,
        save_total_limit=5,
        report_to="wandb"
    )
    data_collator = setup.get_data_collator(tokenizer)

    # Training
    trainer = ModelTrainer(model, tokenizer, dataset, data_collator, training_args)
    stats = trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
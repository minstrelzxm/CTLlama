from setup import SetupCPT
from train import CPTTrainer

def main():
    # 1) Prepare model & data with 4096-token context and LoRA
    setup = SetupCPT(
        model_name="meta-llama/Llama-3.1-8B",
        data_file="data/data.json",
        max_seq_length=4096,
        lora_r=16,
        lora_alpha=16,
    )
    tokenized_ds = setup.prepare_dataset()

    # 2) Train CPT with LoRA adapters
    # Effective batch = per_device_batch_size * grad_accum * num_gpus.
    # Example: 2 * 8 * 8 = 128 (matches README).
    trainer = CPTTrainer(
        model=setup.model,
        tokenizer=setup.tokenizer,
        train_dataset=tokenized_ds,
        output_dir="cpt_output",
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=3,
    )
    trainer.train()

    # 3) Save final CPT model + adapters
    setup.save_tokenizer("cpt_output")
    trainer.save_model("cpt_output")
    print("CPT with LoRA finished — artifacts in cpt_output/")

if __name__ == "__main__":
    main()

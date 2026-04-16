import os

from setup import RLSetup
from train import RLTrainer
from utils import (
    correctness_reward_func, int_reward_func,
    strict_format_reward_func, soft_format_reward_func,
    xmlcount_reward_func
)

def main():
    peft_cfg = {
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"
        ],
        'use_gradient_checkpointing': 'unsloth',
        'random_state': 3407
    }
    setup = RLSetup(
        model_name="minstrelzxm/CTLlama-8b-instruct-distill",
        token=os.environ["HF_TOKEN"],
        peft_config=peft_cfg
    )
    setup.login()
    model, tokenizer = setup.init_model_and_tokenizer()
    dataset = setup.prepare_dataset()
    config = setup.get_training_config(output_dir="output/grpo")
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ]
    trainer = RLTrainer(model, tokenizer, dataset, reward_funcs, config)
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()
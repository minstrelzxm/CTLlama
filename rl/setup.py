import wandb
from unsloth import FastLanguageModel
from datasets import Dataset
from utils import get_gsm8k_questions
from trl import GRPOConfig

class RLSetup:
    def __init__(
        self,
        model_name: str,
        token: str,
        max_seq_length: int = 4096,
        max_prompt_length: int = 512,
        lora_rank: int = 32,
        peft_config: dict = None
    ):
        self.model_name = model_name
        self.token = token
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.lora_rank = lora_rank
        self.peft_config = peft_config or {}

    def login(self):
        wandb.login()

    def init_model_and_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=False,
            fast_inference=False,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.6,
            token=self.token
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=self.peft_config.get('target_modules', []),
            lora_alpha=self.lora_rank,
            use_gradient_checkpointing=self.peft_config.get('use_gradient_checkpointing', 'unsloth'),
            random_state=self.peft_config.get('random_state', 3407)
        )
        return model, tokenizer

    def prepare_dataset(self, split: str = "train") -> Dataset:
        return get_gsm8k_questions(split=split)

    def get_training_config(self, **kwargs) -> GRPOConfig:
        output_dir = kwargs.pop('output_dir', 'output/grpo')
        return GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=6,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_seq_length - self.max_prompt_length,
            max_steps=2000,
            save_steps=300,
            save_total_limit=5,
            max_grad_norm=0.1,
            report_to="wandb",
            output_dir=output_dir,
            skip_special_tokens=False,
            **kwargs
        )
import wandb
import torch
import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import formatting_prompts_func

wandb.login()

max_seq_length = 8192*2 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "minstrelzxm/ctllama-8b-base",
    max_seq_length = max_seq_length,
    dtype = dtype,
    token = "your_only_token_here",
)

tokenizer.padding_side = "right"
tokenizer.pad_token_id = 128004
tokenizer.add_special_tokens({"additional_special_tokens":['<begin_of_ref>', '<end_of_ref>','<think>','</think>', '<answer>', '</answer>']})
new_template = """
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- set loop_messages = messages %}
{%- for message in loop_messages %}
    {%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] | trim + '<|end_of_text|>' %}
        {%- if loop.index0 == 0 %}
            {%- set content = bos_token + content %}
        {%- endif %}{{ content }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ '<|start_header_id|>assistant<|end_header_id|>' }}
{%- endif %}
"""
tokenizer.chat_template = new_template
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens","lm_head"],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def formatting_prompts_func(examples):
    systems = examples['system_prompt']
    instructions = examples["instruction"]
    inputs       = examples["reference"]
    reasonings = examples["modified_reasoning"]
    outputs      = examples["formatted_label"]
    texts = []
    for system, instruction, input, reasoning, output in zip(systems, instructions, inputs, reasonings, outputs):
        messages = [{"role": "system", "content": system},
                   {"role": "user", "content": instruction + "\n" + input},
                   {"role": "assistant", "content": f"<think>{reasoning}</think>" + f"\n" + f"<answer>{output}</answer>"}]
    # apply chat template here
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = False)
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("json", data_files="data/sft_data.json", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_seq_length)

# Apply the tokenization to your dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=3,  # Set to desired number of epochs
    learning_rate=7e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1, # was 20
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_steps=200,
    save_total_limit=5,
    report_to="wandb",  # Use "none" if not reporting to any platform
)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True if using masked language modeling
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,  # Your pre-tokenized dataset
    # dataset_text_field = "text",
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

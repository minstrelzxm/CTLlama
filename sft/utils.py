# utils.py

def formatting_prompts_func(examples, tokenizer):
    systems = examples['system_prompt']
    instructions = examples['instruction']
    inputs = examples['reference']
    reasonings = examples['modified_reasoning']
    outputs = examples['formatted_label']
    texts = []
    for system, instr, inp, reasoning, out in zip(systems, instructions, inputs, reasonings, outputs):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instr + "\n" + inp},
            {"role": "assistant", "content": f"<think>{reasoning}</think>\n<answer>{out}</answer>"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def tokenize_function(examples, tokenizer, max_seq_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length
    )
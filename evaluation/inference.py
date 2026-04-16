"""
Inference entry point for CTLlama.

Loads a (base model + LoRA adapter) checkpoint and generates responses for a
JSONL prompt file. Supports any of the three training stages (CPT, SFT, RL)
as long as the adapter directory contains `adapter_config.json` and the
tokenizer files.

Example:
    python inference.py \\
        --base-model minstrelzxm/ctllama-8b-base \\
        --adapter-dir ../rl/grpo_saved_lora \\
        --prompts prompts.jsonl \\
        --output predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTLlama inference")
    p.add_argument("--base-model", required=True, help="HF repo or local path to base model")
    p.add_argument("--adapter-dir", required=True, help="Directory with LoRA adapter + tokenizer")
    p.add_argument("--prompts", required=True, help="JSONL file with a 'prompt' or 'messages' field per line")
    p.add_argument("--output", required=True, help="JSONL file to write predictions to")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_model(base_model: str, adapter_dir: str, device: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    # Adapter may have resized embeddings (special tokens added during SFT).
    if len(tokenizer) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def iter_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_prompt(tokenizer, record: dict) -> str:
    if "messages" in record:
        return tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=True
        )
    return record["prompt"]


@torch.inference_mode()
def generate(model, tokenizer, text: str, args) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.base_model, args.adapter_dir, args.device, args.dtype)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in iter_prompts(args.prompts):
            text = render_prompt(tokenizer, record)
            response = generate(model, tokenizer, text, args)
            record["response"] = response
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

# CTLlama — Fixes Required

Review of the CPT, SFT, and GRPO RL pipelines. Items are grouped by severity and ordered by priority within each group.

**Status:** 16 / 16 resolved across 5 iterations. `pyflakes` clean, all files parse. Draft folders untouched.

---

## CRITICAL — Pipeline will not run

### 1. CPT uses masked-LM collator on a causal model ✅ FIXED (iter 1)

- **File:** [cpt/train.py:22-26](cpt/train.py#L22-L26), [cpt/run.py:24](cpt/run.py#L24)
- **Problem:** `DataCollatorForLanguageModeling(mlm=True)` requires a `[MASK]` token. LLaMA-3.1 has none, so training will crash with `ValueError: This tokenizer does not have a mask token...`. CPT on a causal model must use autoregressive next-token prediction, not MLM.
- **Fix applied:** Collator → `mlm=False`. `mlm_probability` dropped from `CPTTrainer.__init__` and [cpt/run.py](cpt/run.py). [README.md:56](README.md#L56) updated to "Causal (autoregressive) language modeling".

### 2. CPT `lora_target_modules` is a tuple, not a list ✅ FIXED (iter 1)

- **File:** [cpt/setup.py:28-32](cpt/setup.py#L28-L32)
- **Problem:** A trailing comma after the closing bracket wraps the list in a tuple: `lora_target_modules == ([...],)`. `LoraConfig` will either raise or silently apply no LoRA.
- **Fix applied:** Removed the trailing comma and re-indented for clarity.

### 3. RL imports a module that does not exist ✅ FIXED (iter 2)

- **File:** [rl/setup.py:4](rl/setup.py#L4)
- **Problem:** `from utils_rl import get_gsm8k_questions` — there is no `utils_rl.py` in [rl/](rl/). The function lives in `utils.py`.
- **Fix applied:** `from utils import get_gsm8k_questions`.

### 4. RL `get_training_config` passes `output_dir` twice ✅ FIXED (iter 2)

- **File:** [rl/setup.py:70-72](rl/setup.py#L70-L72)
- **Problem:** `output_dir=kwargs.get(...)` pulls it out, then `**kwargs` re-passes it, causing `TypeError: got multiple values for keyword argument 'output_dir'`.
- **Fix applied:** `kwargs.pop('output_dir', 'output/grpo')` before `GRPOConfig` is constructed.

### 5. RL target-module typo `emb_tokens` ✅ FIXED (iter 2)

- **File:** [rl/run.py:15](rl/run.py#L15)
- **Problem:** LLaMA's embedding module is `embed_tokens`. `"emb_tokens"` matches nothing — PEFT will either raise or skip it, leaving the new special tokens (`<think>`, `<answer>`, …) un-adapted in RL.
- **Fix applied:** `"emb_tokens"` → `"embed_tokens"`.

---

## HIGH — Silently wrong behavior

### 6. SFT computes loss on the entire sequence, not just the response ✅ FIXED (iter 3)

- **File:** [sft/setup.py:73-77](sft/setup.py#L73-L77), [sft/utils.py](sft/utils.py)
- **Problem:** `DataCollatorForLanguageModeling(mlm=False)` copies `input_ids` into `labels` verbatim. Loss is minimized over system prompt + user message + response.
- **Fix applied:** Switched to `trl.DataCollatorForCompletionOnlyLM` with response template `<|start_header_id|>assistant<|end_header_id|>\n\n`. Only assistant-response tokens contribute to loss.

### 7. SFT pads every example to 16,384 tokens ✅ FIXED (iter 3)

- **File:** [sft/setup.py:13](sft/setup.py#L13), [sft/utils.py:21-27](sft/utils.py#L21-L27)
- **Problem:** `max_seq_length = 8192 * 2` combined with `padding="max_length"` forces every short QA pair to 16K tokens.
- **Fix applied:** Dropped `padding="max_length"` from `tokenize_function`. The collator now pads per-batch to the longest sample.

### 8. CPT tokenizer has no `pad_token` ✅ FIXED (iter 1)

- **File:** [cpt/setup.py:35](cpt/setup.py#L35)
- **Problem:** LLaMA-3 tokenizer ships without a `pad_token`. The collator will fail when it tries to pad.
- **Fix applied:** Added `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token` immediately after tokenizer load.

### 9. `int_reward_func` rejects valid numeric answers ✅ FIXED (iter 2)

- **File:** [rl/utils.py:42-45](rl/utils.py#L42-L45)
- **Problem:** `str.isdigit()` returns `False` for `"-42"`, `"3.14"`, `"1,000"`. GSM8K has occasional negative or decimal answers.
- **Fix applied:** New `_is_numeric` helper using `float()` after stripping commas. `int_reward_func` now uses it.

---

## MEDIUM — Quality and hygiene

### 10. SFT hard-codes pad token id ✅ FIXED (iter 3)

- **File:** [sft/setup.py:41](sft/setup.py#L41)
- **Problem:** `tokenizer.pad_token_id = 128004` silently breaks on any non-LLaMA-3 tokenizer.
- **Fix applied:** Pad is now set via the token string `<|finetune_right_pad_id|>` when present in vocab, else `eos_token`. Only runs when `pad_token is None`.

### 11. CPT batch size likely OOMs ✅ FIXED (iter 1)

- **File:** [cpt/run.py:21](cpt/run.py#L21)
- **Problem:** `per_device_batch_size=16` at seq length 4096 on an 8B model with no grad checkpointing or dtype specified.
- **Fix applied:** `per_device_batch_size=2`, `gradient_accumulation_steps=8`, `bf16=True`, `gradient_checkpointing=True` in `TrainingArguments`. Base model loads with `torch_dtype=torch.bfloat16`.

### 12. README batch size disagrees with code (CPT) ✅ FIXED (iter 1)

- **File:** [README.md:59](README.md#L59) vs [cpt/run.py:21](cpt/run.py#L21)
- **Problem:** README said "Batch size: 128", code used 16 with no grad-accum.
- **Fix applied:** README now says "Effective batch size: 128 (per-device 2 × grad-accum 8 × 8 GPUs)".

### 13. README LoRA rank disagrees with code (SFT) ✅ FIXED (iter 3)

- **File:** [README.md:65](README.md#L65) vs [sft/run.py:9](sft/run.py#L9)
- **Problem:** README claimed `rank=32, α=32`; code used `r=16, lora_alpha=32`.
- **Fix applied:** Bumped code to `r=32` to match README.

### 14. Placeholder HF token committed to source ✅ FIXED (iter 4)

- **File:** [sft/run.py:25](sft/run.py#L25), [rl/run.py:22](rl/run.py#L22)
- **Problem:** `token="your_only_token_here"` would 401 on HF Hub and risked real-token leakage into git.
- **Fix applied:** Both scripts now read `os.environ["HF_TOKEN"]`. Added [.env.example](.env.example) and `.env` to [.gitignore](.gitignore).

### 15. RL trainer does not save tokenizer ✅ FIXED (iter 2)

- **File:** [rl/train.py:26-27](rl/train.py#L26-L27)
- **Problem:** `save_lora` writes adapter weights only. Special tokens added in SFT (`<think>`, `<answer>`, …) wouldn't round-trip.
- **Fix applied:** `save()` now also calls `self.tokenizer.save_pretrained(output_dir)`.

### 16. `evaluation/inference.py` is empty ✅ FIXED (iter 4)

- **File:** [evaluation/inference.py](evaluation/inference.py)
- **Problem:** 0-byte file. The pipeline's evaluation stage was missing.
- **Fix applied:** Implemented CLI (argparse), base+adapter loading via `PeftModel.from_pretrained`, JSONL prompt/response IO, and chat-template support via `tokenizer.apply_chat_template`. Re-sizes embeddings on load if the adapter added tokens.

---

## Bonus cleanups (iteration 5 — proactive polish)

- Removed 8 unused imports flagged by pyflakes ([sft/run.py](sft/run.py), [sft/setup.py](sft/setup.py), [rl/run.py](rl/run.py), [rl/setup.py](rl/setup.py)).
- Removed dead `stats = trainer.train()` assignments in [sft/run.py](sft/run.py) and [rl/run.py](rl/run.py).
- Dropped `special_tokens_mask` from [cpt/utils.py](cpt/utils.py) — only needed for MLM, and CPT is now causal.
- Expanded [.gitignore](.gitignore) to exclude training artifacts (`cpt_output/`, `output/`, `grpo_saved_lora/`, `lora_model/`, `wandb/`), Python caches, and `.env*`.
- Added [.env.example](.env.example) documenting the required `HF_TOKEN` and optional `WANDB_API_KEY`.

---

## Fix order followed

1. **CPT blockers** — items 1, 2, 8, then 11, 12 ✅
2. **RL blockers** — items 3, 4, 5, then 9, 15 ✅
3. **SFT correctness** — items 6, 7, then 10, 13 ✅
4. **Cleanup** — items 14, 16 ✅
5. **Polish** (proactive) — unused imports, .env, dead code, expanded .gitignore ✅

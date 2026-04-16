# CTLlama — Fixes Required

Review of the CPT, SFT, and GRPO RL pipelines. Items are grouped by severity and ordered by priority within each group.

---

## CRITICAL — Pipeline will not run

### 1. CPT uses masked-LM collator on a causal model
- **File:** [cpt/train.py:22-26](cpt/train.py#L22-L26), [cpt/run.py:24](cpt/run.py#L24)
- **Problem:** `DataCollatorForLanguageModeling(mlm=True)` requires a `[MASK]` token. LLaMA-3.1 has none, so training will crash with `ValueError: This tokenizer does not have a mask token...`. CPT on a causal model must use autoregressive next-token prediction, not MLM.
- **Fix:** Set `mlm=False` in the collator. Drop `mlm_probability` from [cpt/run.py](cpt/run.py) and [cpt/train.py](cpt/train.py). Update [README.md:56](README.md#L56) — "Masked-language modeling" is incorrect for CPT on LLaMA.

### 2. CPT `lora_target_modules` is a tuple, not a list
- **File:** [cpt/setup.py:28-32](cpt/setup.py#L28-L32)
- **Problem:** A trailing comma after the closing bracket wraps the list in a tuple: `lora_target_modules == ([...],)`. `LoraConfig` will either raise or silently apply no LoRA.
- **Fix:** Remove the trailing comma after `]`.

### 3. RL imports a module that does not exist
- **File:** [rl/setup.py:4](rl/setup.py#L4)
- **Problem:** `from utils_rl import get_gsm8k_questions` — there is no `utils_rl.py` in [rl/](rl/). The function lives in `utils.py`.
- **Fix:** Change to `from utils import get_gsm8k_questions`.

### 4. RL `get_training_config` passes `output_dir` twice
- **File:** [rl/setup.py:70-72](rl/setup.py#L70-L72)
- **Problem:** `output_dir=kwargs.get(...)` pulls it out, then `**kwargs` re-passes it, causing `TypeError: got multiple values for keyword argument 'output_dir'`.
- **Fix:** Use `kwargs.pop('output_dir', 'output/grpo')` before constructing `GRPOConfig`.

### 5. RL target-module typo `emb_tokens`
- **File:** [rl/run.py:15](rl/run.py#L15)
- **Problem:** LLaMA's embedding module is `embed_tokens`. `"emb_tokens"` matches nothing — PEFT will either raise or skip it, leaving the new special tokens (`<think>`, `<answer>`, …) un-adapted in RL.
- **Fix:** Change to `"embed_tokens"`.

---

## HIGH — Silently wrong behavior

### 6. SFT computes loss on the entire sequence, not just the response
- **File:** [sft/setup.py:73-77](sft/setup.py#L73-L77), [sft/utils.py](sft/utils.py)
- **Problem:** `DataCollatorForLanguageModeling(mlm=False)` copies `input_ids` into `labels` verbatim. Loss is minimized over system prompt + user message + response — the model is rewarded for memorizing boilerplate. This contradicts [README.md:64](README.md#L64) which claims loss is only on CoT + answer.
- **Fix:** Use TRL's `DataCollatorForCompletionOnlyLM` with the assistant-response template, or manually set `labels = -100` for prompt tokens before the response span.

### 7. SFT pads every example to 16,384 tokens
- **File:** [sft/setup.py:13](sft/setup.py#L13), [sft/utils.py:21-27](sft/utils.py#L21-L27)
- **Problem:** `max_seq_length = 8192 * 2` combined with `padding="max_length"` forces every short QA pair to 16K tokens. Massive memory and throughput waste.
- **Fix:** Set `padding=False` in `tokenize_function` — the collator pads per-batch to the longest sample automatically. Also consider lowering `max_seq_length` to 4096 or 8192 unless examples genuinely exceed that.

### 8. CPT tokenizer has no `pad_token`
- **File:** [cpt/setup.py:35](cpt/setup.py#L35)
- **Problem:** LLaMA-3 tokenizer ships without a `pad_token`. The collator will fail when it tries to pad.
- **Fix:** After loading:
  ```python
  if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
  ```
  Or use `128004` (`<|finetune_right_pad_id|>`) consistent with SFT.

### 9. `int_reward_func` rejects valid numeric answers
- **File:** [rl/utils.py:42-45](rl/utils.py#L42-L45)
- **Problem:** `str.isdigit()` returns `False` for `"-42"`, `"3.14"`, `"1,000"`. GSM8K has occasional negative or decimal answers, and model output may contain commas.
- **Fix:**
  ```python
  def _is_numeric(s: str) -> bool:
      try:
          float(s.replace(",", ""))
          return True
      except ValueError:
          return False
  ```

---

## MEDIUM — Quality and hygiene

### 10. SFT hard-codes pad token id
- **File:** [sft/setup.py:41](sft/setup.py#L41)
- **Problem:** `tokenizer.pad_token_id = 128004` silently breaks on any non-LLaMA-3 tokenizer.
- **Fix:** Use the token string — `tokenizer.pad_token = "<|finetune_right_pad_id|>"` — or fall back to `eos_token`.

### 11. CPT batch size likely OOMs
- **File:** [cpt/run.py:21](cpt/run.py#L21)
- **Problem:** `per_device_batch_size=16` at seq length 4096 on an 8B model with no grad checkpointing, no quantization, no specified dtype. Won't fit on any consumer GPU.
- **Fix:** Add `gradient_checkpointing=True`, `bf16=True`, and either reduce the per-device batch or add `gradient_accumulation_steps` to [cpt/train.py](cpt/train.py).

### 12. README batch size disagrees with code (CPT)
- **File:** [README.md:59](README.md#L59) vs [cpt/run.py:21](cpt/run.py#L21)
- **Problem:** README says "Batch size: 128", code uses 16 with no grad-accum.
- **Fix:** Document effective batch (with grad accum) or reconcile the numbers.

### 13. README LoRA rank disagrees with code (SFT)
- **File:** [README.md:65](README.md#L65) vs [sft/run.py:9](sft/run.py#L9)
- **Problem:** README claims `rank=32, α=32`; code uses `r=16, lora_alpha=32`.
- **Fix:** Pick one and make both match.

### 14. Placeholder HF token committed to source
- **File:** [sft/run.py:25](sft/run.py#L25), [rl/run.py:22](rl/run.py#L22)
- **Problem:** `token="your_only_token_here"` will 401 on HF Hub. Worse, if someone replaces it with a real token it will leak into git history.
- **Fix:**
  ```python
  import os
  token = os.environ["HF_TOKEN"]
  ```
  Add `HF_TOKEN` to `.env` (already covered by `.gitignore`? confirm).

### 15. RL trainer does not save tokenizer
- **File:** [rl/train.py:26-27](rl/train.py#L26-L27)
- **Problem:** `save_lora` writes adapter weights only. Special tokens added in SFT (`<think>`, `<answer>`, …) won't round-trip unless the tokenizer state is saved alongside.
- **Fix:** Call `self.tokenizer.save_pretrained(output_dir)` in `save()`.

### 16. `evaluation/inference.py` is empty
- **File:** [evaluation/inference.py](evaluation/inference.py)
- **Problem:** 0-byte file. The pipeline's evaluation stage is missing.
- **Fix:** Implement the inference/eval script, or remove the empty file and [evaluation/](evaluation/) directory until it's ready.

---

## Suggested fix order

1. **CPT blockers** — items 1, 2, 8 (then 11, 12)
2. **RL blockers** — items 3, 4, 5 (then 9, 15)
3. **SFT correctness** — items 6, 7 (then 10, 13)
4. **Cleanup** — items 14, 16

The 5 CRITICAL items (1–5) must be fixed before any stage can run end-to-end. Item 6 is the most important silent-correctness bug — SFT will appear to train but learn the wrong objective.

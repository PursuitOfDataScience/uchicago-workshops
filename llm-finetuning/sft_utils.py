"""
SFT (Supervised Fine-Tuning) Utilities for LLM Fine-Tuning Workshop

This module contains all the core functions and classes needed for fine-tuning
Large Language Models using both full fine-tuning and LoRA approaches.

Functions:
    - setup_environment: Configure environment variables and settings
    - get_max_length: Extract max sequence length from model config
    - apply_chat_template: Format messages with chat template
    - build_sft_example: Build a single training example with label masking
    - tokenize_sft_dataset: Tokenize an entire dataset for SFT
    - ensure_input_grads: Enable gradient computation for inputs
    - train_with_auto_batch: Train with automatic batch size reduction on OOM
    - plot_loss: Plot training and validation loss curves

    High-level notebook helpers (keep the notebook clean):
    - load_lima_dataset: Load LIMA and convert to messages format
    - print_dataset_stats: Print dataset statistics and create eval split
    - load_tokenizer: Load and configure a tokenizer
    - show_chat_template_example: Demonstrate chat template formatting
    - prepare_sft_data: Tokenize + wrap datasets for training
    - inspect_training_example: Inspect a single tokenized training example
    - load_model: Load a causal LM in BF16 with SDPA
    - cleanup_memory: Free GPU memory and run garbage collection
    - load_ultrachat_data: Load and expand UltraChat dataset
    - create_ultrachat_datasets: Create UltraChatDataset objects with verification
    - inspect_ultrachat_example: Inspect a raw UltraChat conversation
    - build_ultrachat_training_args: Build TrainingArguments for UltraChat-style training
    - create_and_run_ultrachat_trainer: Create SFTTrainer and run training
    - print_model_architecture: Print model layer structure
    - configure_lora: Apply LoRA adapters and report parameter counts
    - print_comparison_table: Print Full FT vs LoRA comparison

Classes:
    - SFTDataset: PyTorch Dataset wrapper for SFT training
    - UltraChatDataset: On-the-fly tokenization dataset for UltraChat
    - SFTCollator: Data collator for variable-length sequences
    - GenerationCallback: Generate samples during training (for LIMA)
    - UltraChatCallback: Generate samples during UltraChat training
    - SFTTrainer: Custom trainer with proper loss computation
"""

import os
import math
import gc
import textwrap
import warnings
import statistics
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, PreTrainedTokenizerBase
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model


# =============================================================================
# Environment Setup
# =============================================================================
def setup_environment():
    """
    Configure environment variables and PyTorch settings for optimal training.
    
    Sets up:
    - TOKENIZERS_PARALLELISM: Disabled to avoid deadlocks with multiprocessing
    - PYTORCH_CUDA_ALLOC_CONF: Enables expandable memory segments for better OOM handling
    - TF32 precision: Enables faster matmul on Ampere+ GPUs (A100, H100)
    - Warnings: Suppressed for cleaner output
    - Jupyter: Text wrapping to prevent horizontal scrolling
    
    Returns:
        dict: Configuration summary with GPU info
    """
    # Set environment variables
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Enable TF32 for faster computation on modern GPUs
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    # Configure Jupyter output to prevent horizontal scrolling
    try:
        from IPython.display import display, HTML
        display(HTML("""
        <style>
            .output_text, .output_stdout, .output_stderr, .output_subarea {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                overflow-x: hidden !important;
                max-width: 100% !important;
            }
            pre {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }
        </style>
        """))
    except ImportError:
        pass  # Not in Jupyter environment
    
    # Set pandas display options if available
    try:
        import pandas as pd
        pd.set_option('display.max_colwidth', 80)
        pd.set_option('display.width', None)
    except ImportError:
        pass
    
    # Gather GPU info
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "memory_gb": props.total_memory / 1024**3
            })
    
    return {"gpus": gpu_info, "cuda_available": torch.cuda.is_available()}


def print_gpu_info():
    """Print GPU hardware information."""
    print("=" * 50)
    print("GPU Hardware Check")
    print("=" * 50)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("No GPU available - training will be slow.")
    
    print("=" * 50)


# =============================================================================
# Global State
# =============================================================================
TRUNCATION_STATS = {"total": 0, "truncated": 0}


def reset_truncation_stats():
    """Reset the global truncation statistics."""
    global TRUNCATION_STATS
    TRUNCATION_STATS = {"total": 0, "truncated": 0}


def get_truncation_stats():
    """Get the current truncation statistics."""
    return TRUNCATION_STATS.copy()


# =============================================================================
# Configuration Utilities
# =============================================================================
def get_max_length(model_or_config) -> int:
    """
    Extract maximum sequence length from model config.
    
    Args:
        model_or_config: A model or config object
        
    Returns:
        Maximum sequence length supported by the model
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    for attr in ["max_position_embeddings", "max_sequence_length", "n_positions"]:
        if hasattr(cfg, attr) and getattr(cfg, attr):
            return int(getattr(cfg, attr))
    return 2048


# =============================================================================
# Chat Template Utilities
# =============================================================================
def apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool = False) -> str:
    """
    Format messages with chat template, with fallback for models without one.
    
    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content' keys
        add_generation_prompt: Whether to add a generation prompt at the end
        
    Returns:
        Formatted string ready for tokenization
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    
    # Fallback format for models without chat template
    bos = getattr(tokenizer, "bos_token", "") or ""
    eos = getattr(tokenizer, "eos_token", "") or ""
    lines = [bos] if bos else []
    for msg in messages:
        lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        lines.append("assistant:")
    return "\n".join(lines) + (eos if eos else "")


# =============================================================================
# Data Processing Functions
# =============================================================================
def build_sft_example(tokenizer, messages: List[Dict], assistant_idx: int, max_length: int) -> Dict:
    """
    Build one training example from an assistant message.
    
    Creates labels that mask out user messages (-100) so only assistant tokens 
    contribute to the loss during training.
    
    Args:
        tokenizer: The tokenizer to use
        messages: Full conversation as list of message dicts
        assistant_idx: Index of the assistant message to train on
        max_length: Maximum sequence length
        
    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels'
    """
    global TRUNCATION_STATS
    
    context = messages[:assistant_idx]
    full_messages = context + [messages[assistant_idx]]
    full_text = apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
    
    # Tokenize - use add_special_tokens=False because chat template already includes BOS/EOS
    encoded = tokenizer(
        full_text, truncation=True, max_length=max_length, 
        padding="max_length", return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    
    # Track truncation
    full_ids = tokenizer(full_text, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        TRUNCATION_STATS["truncated"] += 1
    TRUNCATION_STATS["total"] += 1
    
    # Create label mask (find where assistant content starts)
    seq_len = int(attention_mask.sum().item())
    plain_ids = input_ids[:seq_len].tolist()
    
    stub_messages = context + [{"role": "assistant", "content": ""}]
    stub_text = apply_chat_template(tokenizer, stub_messages, add_generation_prompt=False)
    stub_ids = tokenizer(stub_text, truncation=True, max_length=max_length, padding=False, add_special_tokens=False)["input_ids"]
    
    # Find prefix match
    prefix_len = 0
    for i in range(min(len(plain_ids), len(stub_ids))):
        if plain_ids[i] == stub_ids[i]:
            prefix_len = i + 1
        else:
            break
    
    suffix_len = 0
    for i in range(min(len(plain_ids) - prefix_len, len(stub_ids) - prefix_len)):
        if plain_ids[-(i+1)] == stub_ids[-(i+1)]:
            suffix_len = i + 1
        else:
            break
    
    content_start = prefix_len
    content_end = max(prefix_len, len(plain_ids) - suffix_len)
    
    # Build mask: 1 for assistant tokens, 0 otherwise
    mask = [0] * len(plain_ids)
    for i in range(content_start, content_end):
        mask[i] = 1
    
    # Include EOS if present
    terminators = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
    if content_end < len(plain_ids) and plain_ids[content_end] in terminators:
        mask[content_end] = 1
    
    mask = mask + [0] * (max_length - len(mask))
    mask_tensor = torch.tensor(mask[:max_length], dtype=torch.long)
    
    # Labels: -100 for ignored tokens
    labels = input_ids.clone()
    labels[(mask_tensor == 0) | (attention_mask == 0)] = -100
    
    return {
        "input_ids": input_ids.tolist(), 
        "attention_mask": attention_mask.tolist(), 
        "labels": labels.tolist()
    }


def tokenize_sft_dataset(hf_dataset, tokenizer, max_length: int, num_proc: int = 1):
    """
    Tokenize dataset, creating one example per assistant message.
    
    Args:
        hf_dataset: HuggingFace dataset with 'messages' column
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel tokenization
        
    Returns:
        Tokenized HuggingFace dataset
    """
    def expand_batch(batch):
        all_ids, all_masks, all_labels = [], [], []
        for messages in batch["messages"]:
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    ex = build_sft_example(tokenizer, messages, idx, max_length)
                    all_ids.append(ex["input_ids"])
                    all_masks.append(ex["attention_mask"])
                    all_labels.append(ex["labels"])
        return {"input_ids": all_ids, "attention_mask": all_masks, "labels": all_labels}
    
    print(f"Tokenizing (max_length={max_length})...")
    tokenized = hf_dataset.map(
        expand_batch, batched=True, batch_size=512, num_proc=num_proc,
        remove_columns=hf_dataset.column_names, load_from_cache_file=False, keep_in_memory=True
    )
    print(f"✓ {len(tokenized):,} examples from {len(hf_dataset):,} conversations")
    return tokenized


def expand_conversations(hf_dataset) -> List[Dict]:
    """
    Expand multi-turn conversations: each assistant turn becomes one training example.
    
    Args:
        hf_dataset: HuggingFace dataset with 'messages' column
        
    Returns:
        List of expanded examples
    """
    expanded = []
    for idx in range(len(hf_dataset)):
        messages = hf_dataset[idx].get("messages", [])
        prompt = hf_dataset[idx].get("prompt", None)
        if prompt:
            messages = [{"role": "system", "content": prompt}] + messages
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                expanded.append({"messages": messages[:i + 1], "assistant_idx": i})
    return expanded


# =============================================================================
# Dataset Classes
# =============================================================================
class SFTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for SFT training.
    
    Wraps a pre-tokenized HuggingFace dataset and ensures all sequences
    have exactly max_length tokens.
    """
    
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        ex = self.ds[idx]
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        mask = torch.tensor(ex["attention_mask"], dtype=torch.long)
        labels = torch.tensor(ex["labels"], dtype=torch.long)
        
        # Ensure exact length
        if ids.shape[0] != self.max_length:
            pad_id = self.tokenizer.pad_token_id
            if ids.shape[0] > self.max_length:
                ids = ids[:self.max_length]
                mask = mask[:self.max_length]
                labels = labels[:self.max_length]
            else:
                pad_len = self.max_length - ids.shape[0]
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class UltraChatDataset(torch.utils.data.Dataset):
    """
    On-the-fly tokenization dataset for UltraChat (memory efficient).
    
    Tokenizes examples during __getitem__ rather than pre-tokenizing the entire
    dataset, which saves memory for large datasets.
    """
    
    def __init__(self, expanded: List[Dict], tokenizer, max_len: int):
        self.examples = expanded
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        messages, asst_idx = ex["messages"], ex["assistant_idx"]
        if not messages or asst_idx >= len(messages):
            return {"input_ids": [self.pad_id], "attention_mask": [0], "labels": [-100]}
        
        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        
        # Tokenize full conversation (context + assistant response)
        full_text = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len - 2, 
            padding=False, add_special_tokens=False
        )
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
        
        if bos_id is not None:
            input_ids = [bos_id] + input_ids
            attn_mask = [1] + attn_mask
        if eos_id is not None:
            input_ids = input_ids + [eos_id]
            attn_mask = attn_mask + [1]
        
        # Find prompt length for masking (context only, without assistant response)
        context = messages[:asst_idx]
        prompt_text = apply_chat_template(self.tokenizer, context, add_generation_prompt=True)
        prompt_enc = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_len - 2, 
            padding=False, add_special_tokens=False
        )
        prompt_ids = [bos_id] + prompt_enc["input_ids"] if bos_id else prompt_enc["input_ids"]
        
        # Find where the prompt ends (longest matching prefix)
        prompt_len = 0
        for i in range(min(len(prompt_ids), len(input_ids))):
            if input_ids[i] == prompt_ids[i]:
                prompt_len = i + 1
            else:
                break
        
        # Fallback: use the prompt token length if matching failed
        if prompt_len == 0:
            prompt_len = min(len(prompt_ids), len(input_ids) - 1)
        
        # Ensure we have at least one token to predict
        prompt_len = min(prompt_len, len(input_ids) - 1)
        
        # Labels: -100 for prompt tokens (ignored in loss), actual ids for response
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


@dataclass
class SFTCollator:
    """
    Data collator that pads variable-length sequences for batch training.
    
    Uses left-padding so that actual content is at the right side,
    which is important for causal language model generation.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if len(f["input_ids"]) > 1]
        if not features:
            return {
                "input_ids": torch.tensor([[self.tokenizer.pad_token_id]]),
                "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                "labels": torch.tensor([[-100]])
            }
        
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length or 99999)
        max_len = ((max_len + 7) // 8) * 8  # Round to multiple of 8 for efficiency
        
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, f in enumerate(features):
            n = min(len(f["input_ids"]), max_len)
            # Left padding: place content at the right
            input_ids[i, -n:] = torch.tensor(f["input_ids"][-n:])
            attention_mask[i, -n:] = torch.tensor(f["attention_mask"][-n:])
            labels[i, -n:] = torch.tensor(f["labels"][-n:])
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# =============================================================================
# Trainer Callbacks
# =============================================================================

# ANSI color pairs for alternating Q-A pairs in generation output
# Each pair is (question_color, answer_color) to visually distinguish pairs
_QA_COLORS = [
    ("\033[1;34m", "\033[0;34m"),   # Bold blue / blue
    ("\033[1;32m", "\033[0;32m"),   # Bold green / green
    ("\033[1;35m", "\033[0;35m"),   # Bold magenta / magenta
    ("\033[1;36m", "\033[0;36m"),   # Bold cyan / cyan
    ("\033[1;33m", "\033[0;33m"),   # Bold yellow / yellow
]
_RESET = "\033[0m"


class GenerationCallback(TrainerCallback):
    """
    Generate sample responses during training to monitor quality.
    
    This callback generates responses to a set of test questions at regular
    intervals during training, allowing you to qualitatively assess model improvement.
    """


    QUESTIONS = [
        # Keep one simple baseline
        "Explain what machine learning is in simple terms.",
        
        # Reasoning / slightly challenging
        "Why might correlation not imply causation? Give an example.",
        "What are the trade-offs between model complexity and generalization?",
        
        # Instruction-following
        "Summarize the purpose of regularization in exactly two sentences.",
        
        # Creative but grounded
        "Write a short poem about the ocean.",
    ]
    
    def __init__(self, tokenizer, steps: int = 100, skip_initial: bool = False):
        self.tokenizer = tokenizer
        self.steps = steps
        self._initial_done = skip_initial
    
    def _generate(self, model, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": question}
        ]
        prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, 
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        model.train()
        return response
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width for better display."""
        import textwrap
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            if len(line) > width:
                wrapped_lines.extend(textwrap.wrap(line, width=width))
            else:
                wrapped_lines.append(line)
        return '\n'.join(wrapped_lines)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initial_done and model:
            print("\n" + "=" * 50)
            print("Initial Responses (Before Training)")
            print("=" * 50)
            for i, q in enumerate(self.QUESTIONS, 1):
                q_color, a_color = _QA_COLORS[(i - 1) % len(_QA_COLORS)]
                response = self._wrap_text(self._generate(model, q))
                print(f"\n{q_color}[Q{i}] {self._wrap_text(q)}{_RESET}")
                print(f"{a_color}[A{i}] {response}{_RESET}")
            self._initial_done = True
        return control
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model and state.global_step > 0 and state.global_step % self.steps == 0:
            print(f"\n--- Step {state.global_step} ---")
            for i, q in enumerate(self.QUESTIONS, 1):
                q_color, a_color = _QA_COLORS[(i - 1) % len(_QA_COLORS)]
                response = self._wrap_text(self._generate(model, q))
                print(f"\n{q_color}[Q{i}] {self._wrap_text(q)}{_RESET}")
                print(f"{a_color}[A{i}] {response}{_RESET}")
        return control


class UltraChatCallback(TrainerCallback):
    """
    Generate samples during UltraChat training to monitor quality improvement.
    
    Similar to GenerationCallback but with different formatting and more frequent
    generation intervals suitable for the longer UltraChat training runs.
    """
    
    PROMPTS = [
        # Keep one simple baseline
        "Explain what machine learning is in simple terms.",
        
        # Reasoning / slightly challenging
        "Why might correlation not imply causation? Give an example.",
        "What are the trade-offs between model complexity and generalization?",
        
        # Instruction-following
        "Summarize the purpose of regularization in exactly two sentences.",
        
        # Creative but grounded
        "Write a short poem about the ocean.",
    ]
    
    def __init__(self, tokenizer, generation_steps: int = 50):
        self.tokenizer = tokenizer
        self.generation_steps = generation_steps
        self._initial_done = False
        self._last_step = -1
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width for better display."""
        import textwrap
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            if len(line) > width:
                wrapped_lines.extend(textwrap.wrap(line, width=width))
            else:
                wrapped_lines.append(line)
        return '\n'.join(wrapped_lines)
    
    def _generate_samples(self, model, label: str):
        """Generate responses to test prompts."""
        model.eval()
        device = next(model.parameters()).device
        
        print(f"\n{'=' * 70}")
        print(f"Generation Check: {label}")
        print("=" * 70)
        
        bos_id = self.tokenizer.bos_token_id
        
        for i, prompt in enumerate(self.PROMPTS, 1):
            formatted = f"user: {prompt}\nassistant:"
            ids = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"]
            if bos_id is not None:
                bos_tensor = torch.tensor([[bos_id]], dtype=ids.dtype)
                ids = torch.cat([bos_tensor, ids], dim=1)
            ids = ids.to(device)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            wrapped_prompt = self._wrap_text(prompt)
            wrapped_response = self._wrap_text(response)
            q_color, a_color = _QA_COLORS[(i - 1) % len(_QA_COLORS)]
            print(f"\n{q_color}[Q{i}] {wrapped_prompt}{_RESET}")
            print(f"{a_color}[A{i}] {wrapped_response}{_RESET}")
        
        print("\n" + "-" * 70)
        model.train()
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Generate baseline samples before training."""
        if model and not self._initial_done:
            self._generate_samples(model, "BASELINE (Before SFT)")
            self._initial_done = True
        return control
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate samples at regular intervals during training."""
        step = state.global_step
        if model and step > 0 and step % self.generation_steps == 0 and step != self._last_step:
            self._last_step = step
            self._generate_samples(model, f"Step {step}")
        return control


# =============================================================================
# Custom Trainer
# =============================================================================
class SFTTrainer(Trainer):
    """
    Custom trainer with proper loss computation for SFT.
    
    Ensures that the loss is computed correctly even when the model doesn't
    automatically compute it from the labels.
    
    We set model_accepts_loss_kwargs = False so the Trainer always divides
    loss by gradient_accumulation_steps before backprop. Without this, the
    Trainer skips that division (because the underlying model's forward()
    has **kwargs), which inflates logged training loss by a factor of
    gradient_accumulation_steps.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force the Trainer to always normalize loss by gradient_accumulation_steps.
        # The model's forward() has **kwargs which makes the Trainer think the model
        # handles its own loss scaling — but our compute_loss returns mean-reduced
        # loss, so we need the Trainer's standard division.
        self.model_accepts_loss_kwargs = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs.get("attention_mask"), 
            labels=labels
        )
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                shifted_labels.view(-1), 
                ignore_index=-100
            )
        
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Training Utilities
# =============================================================================
def _unwrap_model(model):
    """Unwrap compiled/wrapped models for saving."""
    return getattr(model, "_orig_mod", None) or getattr(model, "module", None) or model


def ensure_input_grads(model):
    """
    Enable gradient computation for inputs (needed for gradient checkpointing).
    
    Args:
        model: The model to enable input gradients for
    """
    if getattr(model, "_input_grads_enabled", False):
        return
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        model._input_grads_enabled = True


def preprocess_logits(logits, labels):
    """Extract predictions from logits for metrics computation."""
    return (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)


def compute_metrics(eval_pred):
    """Compute loss and perplexity metrics."""
    if hasattr(eval_pred, 'losses') and eval_pred.losses is not None:
        avg_loss = float(np.mean(eval_pred.losses))
        return {"eval_loss": avg_loss, "perplexity": float(np.exp(avg_loss))}
    return {}


def train_with_auto_batch(
    model, 
    tokenizer, 
    train_ds, 
    eval_ds, 
    callback, 
    output_dir: str,
    effective_bs: int = 16, 
    num_epochs: int = 6,
    max_steps: int = -1
):
    """
    Train with automatic batch size reduction on OOM.
    
    Starts with batch_size=16 and halves it on each OOM error, adjusting
    gradient accumulation to maintain the effective batch size.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_ds: Training dataset
        eval_ds: Evaluation dataset
        callback: Training callback
        output_dir: Directory to save the model
        effective_bs: Target effective batch size
        num_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides num_epochs if > 0)
        
    Returns:
        Trained Trainer object
    """
    batch_size, trainer = 16, None
    
    while batch_size >= 1:
        try:
            grad_accum = max(1, math.ceil(effective_bs / batch_size))
            print(f"\nBatch size: {batch_size}, Grad accum: {grad_accum}, Effective: {batch_size * grad_accum}")
            
            if trainer:
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            args = TrainingArguments(
                output_dir=output_dir, 
                num_train_epochs=num_epochs if max_steps <= 0 else 1000,
                max_steps=max_steps,
                per_device_train_batch_size=batch_size, 
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum, 
                gradient_checkpointing=True,
                optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
                learning_rate=1e-5, 
                lr_scheduler_type="cosine", 
                warmup_ratio=0.05,
                max_grad_norm=1.0, 
                weight_decay=0.01, 
                logging_steps=10,
                save_strategy="no", 
                eval_strategy="steps", 
                eval_steps=50,
                bf16=True, 
                dataloader_num_workers=4, 
                dataloader_pin_memory=True,
                remove_unused_columns=False, 
                report_to="none",
            )
            
            if hasattr(model, "config"):
                model.config.use_cache = False
            ensure_input_grads(model)
            
            trainer = Trainer(
                model=model, 
                args=args, 
                train_dataset=train_ds, 
                eval_dataset=eval_ds,
                callbacks=[callback], 
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits
            )
            trainer.label_names = ["labels"]
            
            trainer.train()
            
            print(f"✅ Training complete!")
            return trainer
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                old_bs = batch_size
                batch_size = max(1, batch_size - 4) if batch_size > 4 else batch_size // 2
                if batch_size < 1:
                    raise RuntimeError("OOM even with batch_size=1")
                print(f"OOM with batch_size={old_bs}, reducing to {batch_size}...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
    
    raise RuntimeError("OOM even with batch_size=1")


# =============================================================================
# Visualization
# =============================================================================
def plot_loss(trainer, title: str = "Training Progress"):
    """
    Plot training and validation loss curves.
    
    Args:
        trainer: Trained Trainer object with log history
        title: Title for the plot
    """
    train_losses, train_steps = [], []
    eval_losses, eval_steps = [], []
    
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            train_losses.append(log["loss"])
            train_steps.append(log["step"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
            eval_steps.append(log["step"])
    
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label='Train', color='#2ecc71')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Eval', color='#e67e22')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# High-Level Notebook Helpers
# =============================================================================

def load_lima_dataset(data_path: str):
    """
    Load LIMA dataset from disk and convert to standard chat messages format.

    Args:
        data_path: Path to the saved LIMA dataset on disk

    Returns:
        HuggingFace dataset with 'messages' column
    """
    raw_dataset = load_from_disk(data_path, keep_in_memory=True)
    train_data = raw_dataset["train"]
    print(f"Loaded {len(train_data)} training examples")

    def _convert_to_messages(example):
        messages = []
        for i, content in enumerate(example["conversations"]):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": content})
        return {"messages": messages}

    train_data = train_data.map(
        _convert_to_messages,
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    # Preview a sample conversation
    print("\nSample Conversation:")
    for msg in train_data[0]["messages"][:4]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    return train_data


def print_dataset_stats(train_data, eval_size: int = 20):
    """
    Print dataset statistics and create a small evaluation split.

    Args:
        train_data: HuggingFace dataset with 'messages' column
        eval_size: Number of examples to reserve for evaluation

    Returns:
        eval_data: HuggingFace dataset slice for evaluation
    """
    message_counts = [len(ex["messages"]) for ex in train_data]
    print(f"Total conversations: {len(train_data)}")
    print(f"Avg turns: {statistics.mean(message_counts):.1f}, Range: {min(message_counts)}-{max(message_counts)}")

    eval_data = train_data.select(range(eval_size))
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    return eval_data


def load_tokenizer(model_path: str, padding_side: str = "right"):
    """
    Load and configure a tokenizer with proper padding settings.

    Args:
        model_path: Path to model/tokenizer directory
        padding_side: 'right' for training, 'left' for variable-length batches

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Chat template: {'Available' if tokenizer.chat_template else 'Not available'}")
    return tokenizer


def show_chat_template_example(tokenizer):
    """
    Demonstrate how the chat template formats a conversation before tokenization.

    Args:
        tokenizer: A loaded tokenizer
    """
    sample_messages = [
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "assistant", "content": "To learn, to love, and to leave the world a little better than you found it."},
    ]

    if tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=False,
        )
    else:
        formatted = "\n".join(f"{m['role']}: {m['content']}" for m in sample_messages)

    print("Formatted text (what the model sees):")
    print(formatted)
    print(f"\nTotal tokens: {len(tokenizer(formatted)['input_ids'])}")


def prepare_sft_data(model_path: str, tokenizer, train_data, eval_data, max_length_cap: int = 2048):
    """
    Load model config, tokenize datasets, and wrap them in PyTorch Dataset objects.

    Args:
        model_path: Path to model directory (for config)
        tokenizer: The tokenizer to use
        train_data: Training HuggingFace dataset
        eval_data: Evaluation HuggingFace dataset
        max_length_cap: Cap on max sequence length

    Returns:
        (train_dataset, eval_dataset, max_length)
    """
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = min(get_max_length(model_config), max_length_cap)
    print(f"Using max_length: {max_length}")

    reset_truncation_stats()
    train_tokenized = tokenize_sft_dataset(train_data, tokenizer, max_length)
    eval_tokenized = tokenize_sft_dataset(eval_data, tokenizer, max_length)

    train_dataset = SFTDataset(train_tokenized, tokenizer, max_length)
    eval_dataset = SFTDataset(eval_tokenized, tokenizer, max_length)

    print(f"Train: {len(train_dataset):,} examples, Eval: {len(eval_dataset):,} examples")
    return train_dataset, eval_dataset, max_length


def inspect_training_example(train_dataset, train_data, tokenizer):
    """
    Inspect a single training example: shapes, masking, and decoded content.

    Args:
        train_dataset: SFTDataset (tokenized)
        train_data: Original HuggingFace dataset with 'messages' column
        tokenizer: The tokenizer used for decoding
    """
    sample = train_dataset[0]

    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Non-padding tokens: {(sample['attention_mask'] == 1).sum().item()}")
    print(f"Tokens with loss (assistant only): {(sample['labels'] != -100).sum().item()}")

    original_messages = train_data[0]["messages"]
    print("\n" + "=" * 60)
    print("COMPLETE TRAINING EXAMPLE")
    print("=" * 60)

    for msg in original_messages:
        role = msg["role"].upper()
        content = msg["content"]
        wrapped_content = "\n".join(
            textwrap.fill(line, width=80) if line.strip() else line
            for line in content.split("\n")
        )
        print(f"\n[{role}]:")
        print("-" * 40)
        print(wrapped_content)
        print("-" * 40)


def load_model(model_path: str, local_files_only: bool = False):
    """
    Load a causal LM in BF16 with SDPA attention and report parameter count.

    Args:
        model_path: Path to model directory
        local_files_only: If True, don't attempt downloads

    Returns:
        Loaded model (on GPU if available)
    """
    device_map = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map or "auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    return model


def load_model_for_ultrachat(model_path: str):
    """
    Load model and tokenizer configured for UltraChat-style training (left padding).

    Args:
        model_path: Path to model directory

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model, tokenizer


def cleanup_memory(*objects, msg: str = "Memory freed."):
    """
    Delete objects, clear GPU cache, and run garbage collection.

    Args:
        *objects: Variable names are not accessible; caller should ``del`` them.
                  This function just handles cache clearing and gc.
        msg: Message to print after cleanup
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(msg)


def load_ultrachat_data(data_path: str, eval_size: int = 100):
    """
    Load UltraChat train/eval splits and expand multi-turn conversations.

    Args:
        data_path: Path to saved UltraChat dataset directory
        eval_size: Number of eval examples to keep

    Returns:
        (train_raw, eval_raw, train_expanded, eval_expanded)
    """
    train_raw = load_from_disk(os.path.join(data_path, "train_sft"))
    eval_raw = load_from_disk(
        os.path.join(data_path, "test_gen")
    ).select(range(eval_size))

    train_expanded = expand_conversations(train_raw)
    eval_expanded = expand_conversations(eval_raw)

    print(f"Train: {len(train_expanded):,} examples, Eval: {len(eval_expanded):,} examples")
    return train_raw, eval_raw, train_expanded, eval_expanded


def create_ultrachat_datasets(train_expanded, eval_expanded, tokenizer, max_ctx: int):
    """
    Create UltraChatDataset objects and verify label masking.

    Args:
        train_expanded: Expanded training examples
        eval_expanded: Expanded evaluation examples
        tokenizer: Tokenizer to use
        max_ctx: Max context length

    Returns:
        (train_ds, eval_ds)
    """
    train_ds = UltraChatDataset(train_expanded, tokenizer, max_ctx)
    eval_ds = UltraChatDataset(eval_expanded, tokenizer, max_ctx)
    print(f"Datasets ready: train={len(train_ds):,}, eval={len(eval_ds):,}")

    sample = train_ds[0]
    num_tokens = len(sample["input_ids"])
    num_labels = sum(1 for l in sample["labels"] if l != -100)
    print(f"\nSample check: {num_tokens} tokens, {num_labels} with loss ({100*num_labels/num_tokens:.1f}%)")
    return train_ds, eval_ds


def inspect_ultrachat_example(raw_dataset, idx: int = 0):
    """
    Print a complete conversation from the raw UltraChat dataset.

    Args:
        raw_dataset: Raw HuggingFace UltraChat dataset
        idx: Index of the example to inspect
    """
    sample_conversation = raw_dataset[idx]["messages"]

    print("=" * 60)
    print("COMPLETE ULTRACHAT TRAINING EXAMPLE")
    print("=" * 60)

    for i, msg in enumerate(sample_conversation):
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 1500:
            content = content[:1500] + "\n... [truncated for display] ..."
        wrapped_content = "\n".join(
            textwrap.fill(line, width=80) if line.strip() else line
            for line in content.split("\n")
        )
        print(f"\n[TURN {i+1} - {role}]:")
        print("-" * 40)
        print(wrapped_content)
        print("-" * 40)

    print(f"\nTotal turns in this conversation: {len(sample_conversation)}")


def build_ultrachat_training_args(
    output_dir: str,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    eval_steps: int = 50,
    logging_steps: int = 10,
):
    """
    Build TrainingArguments for UltraChat-style fine-tuning.

    Args:
        output_dir: Checkpoint / output directory
        max_steps: Total optimisation steps
        batch_size: Per-device training batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup steps
        eval_steps: Evaluate every N steps
        logging_steps: Log loss every N steps

    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        batch_eval_metrics=True,
    )


def create_and_run_ultrachat_trainer(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    training_args,
    max_ctx: int,
    generation_steps: int = 100,
):
    """
    Create an SFTTrainer with UltraChatCallback, run training, and return the trainer.

    On OOM, reduces per-device batch size by 4 (minimum 1) and increases
    gradient accumulation to keep the effective batch size constant, then retries.

    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_ds: Training UltraChatDataset
        eval_ds: Evaluation UltraChatDataset
        training_args: TrainingArguments
        max_ctx: Max context length (for SFTCollator)
        generation_steps: Generate samples every N steps

    Returns:
        Trained SFTTrainer
    """
    if hasattr(model, "config"):
        model.config.use_cache = False

    ensure_input_grads(model)

    batch_size = training_args.per_device_train_batch_size
    grad_accum = training_args.gradient_accumulation_steps
    effective_bs = batch_size * grad_accum
    trainer = None

    while batch_size >= 1:
        try:
            # Rebuild training args with adjusted batch size / grad accum
            adjusted_args = TrainingArguments(
                **{
                    **training_args.to_dict(),
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                }
            )

            if trainer is not None:
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            trainer = SFTTrainer(
                model=model,
                args=adjusted_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=SFTCollator(tokenizer, max_ctx),
                callbacks=[
                    UltraChatCallback(tokenizer, generation_steps=generation_steps)
                ],
            )
            trainer.label_names = ["labels"]

            print(f"Training with batch_size={batch_size}, grad_accum={grad_accum}, effective={batch_size * grad_accum}")
            trainer.train()
            print(f"\nTraining complete! Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")
            return trainer

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                old_bs = batch_size
                batch_size = max(1, batch_size - 4) if batch_size > 4 else batch_size // 2
                if batch_size < 1:
                    raise RuntimeError("OOM even with batch_size=1")
                grad_accum = max(1, math.ceil(effective_bs / batch_size))
                print(f"OOM with batch_size={old_bs}, reducing to {batch_size} (grad_accum={grad_accum})")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    raise RuntimeError("OOM even with batch_size=1")


def print_model_architecture(model):
    """
    Print high-level model architecture (type, layer count, layer-0 structure).

    Args:
        model: A causal LM
    """
    print("=" * 50)
    print("Model Architecture")
    print("=" * 50)
    print(f"Model type: {model.__class__.__name__}")
    print(f"Total layers: {model.config.num_hidden_layers}")
    print(f"\nLayer 0 structure (all layers follow this pattern):")
    print(model.model.layers[0])
    print(f"\n... layers 1-{model.config.num_hidden_layers - 1} have the same structure ...")


def configure_lora(model, target_modules: list, r: int = 64, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """
    Apply LoRA adapters to specified projections and report parameter counts.

    Args:
        model: Base causal LM
        target_modules: List of module names to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers

    Returns:
        Model with LoRA adapters applied
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable_params / total_params
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({pct:.2f}% of total)")
    return model


def print_comparison_table(
    lora_model,
    ultrachat_max_steps: int,
    ultrachat_batch_size: int,
    ultrachat_grad_accum: int,
    ultrachat_ft_trainer=None,
    lora_trainer=None,
    lora_batch_size: int = 4,
    lora_grad_accum: int = 4,
    lora_learning_rate: float = 1e-4,
):
    """
    Print a comparison table between Full Fine-Tuning and LoRA.

    Args:
        lora_model: The LoRA model (to compute param counts)
        ultrachat_max_steps: Training steps used
        ultrachat_batch_size: Per-device batch size for full FT
        ultrachat_grad_accum: Gradient accumulation steps for full FT
        ultrachat_ft_trainer: Trainer from Section 5b (may be None)
        lora_trainer: Trainer from Section 6 (may be None)
        lora_batch_size: Per-device batch size for LoRA
        lora_grad_accum: Gradient accumulation steps for LoRA
        lora_learning_rate: Learning rate for LoRA
    """
    print("=" * 60)
    print("UltraChat Training Results Comparison")
    print("=" * 60)

    lora_total_params = sum(p.numel() for p in lora_model.parameters())
    lora_trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    lora_pct = 100 * lora_trainable_params / lora_total_params

    full_ft_memory_gb = lora_total_params * 2 / 1e9
    lora_memory_gb = lora_total_params * 2 / 1e9

    print(f"""
Training Configuration:
  - Dataset: UltraChat 200K
  - Max steps: {ultrachat_max_steps}

                      | Full FT (5b)  | LoRA (6)
----------------------|---------------|-------------
Learning rate         | 2e-5          | {lora_learning_rate}
Batch size            | {ultrachat_batch_size}            | {lora_batch_size}
Gradient accumulation | {ultrachat_grad_accum}             | {lora_grad_accum}
Effective batch size  | {ultrachat_batch_size * ultrachat_grad_accum}            | {lora_batch_size * lora_grad_accum}
Trainable params      | 100%          | {lora_pct:.2f}%
Trainable count       | {lora_total_params:,} | {lora_trainable_params:,}
Est. param memory     | ~{full_ft_memory_gb * 6:.1f} GB   | ~{lora_memory_gb * 1.5:.1f} GB
Multi-task support    | No            | Yes (swap adapters)

(Note: Full FT needs ~6x model size for optimizer states; LoRA ~1.5x)
""")

    print("Final Evaluation Losses:")
    try:
        if ultrachat_ft_trainer is not None and hasattr(ultrachat_ft_trainer, 'state') and ultrachat_ft_trainer.state.best_metric:
            print(f"  Full FT (5b): {ultrachat_ft_trainer.state.best_metric:.4f}")
    except (NameError, AttributeError):
        print("  Full FT (5b): (trainer not available - run Section 5b first)")

    try:
        if lora_trainer is not None and lora_trainer.state.best_metric:
            print(f"  LoRA (6):     {lora_trainer.state.best_metric:.4f}")
    except (NameError, AttributeError):
        print("  LoRA (6):     (trainer not available - run Section 6 first)")

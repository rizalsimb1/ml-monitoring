"""LoRA/QLoRA fine-tuning trainer using HuggingFace PEFT + TRL."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    base_model: str
    dataset: str
    output_dir: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    target_modules: list = None


class LoRATrainer:
    """High-level interface for LoRA/QLoRA fine-tuning."""

    def __init__(self, **kwargs):
        self.cfg = TrainingConfig(**kwargs)
        self._model = None
        self._tokenizer = None

    def train(self) -> None:
        """Run the full fine-tuning pipeline."""
        print(f"Loading base model: {self.cfg.base_model}")
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()
        model = self._apply_lora(self._model)

        print(f"Loading dataset: {self.cfg.dataset}")
        train_ds = self._load_dataset()

        trainer = self._build_trainer(model, train_ds)
        print("Starting training...")
        trainer.train()

        # Save LoRA adapter weights
        model.save_pretrained(self.cfg.output_dir)
        self._tokenizer.save_pretrained(self.cfg.output_dir)
        print(f"LoRA adapters saved to: {self.cfg.output_dir}")

    def merge_and_save(self, output_path: str) -> None:
        """Merge LoRA weights into base model and save as full model."""
        from peft import PeftModel
        import torch
        print(f"Merging LoRA weights into base model...")
        base = self._load_model(quantized=False)
        merged = PeftModel.from_pretrained(base, self.cfg.output_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(output_path, safe_serialization=True)
        self._tokenizer.save_pretrained(output_path)
        print(f"Merged model saved to: {output_path}")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, quantized: bool = None):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        use_quant = self.cfg.use_4bit if quantized is None else quantized
        bnb_config = None
        if use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=self.cfg.use_nested_quant,
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if not use_quant else "eager",
        )
        model.config.use_cache = False
        return model

    def _apply_lora(self, model):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if self.cfg.use_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.cfg.target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    def _load_dataset(self):
        from datasets import load_dataset
        if self.cfg.dataset.endswith(".jsonl") or self.cfg.dataset.endswith(".json"):
            ds = load_dataset("json", data_files=self.cfg.dataset, split="train")
        else:
            ds = load_dataset(self.cfg.dataset, split="train")
        return ds

    def _build_trainer(self, model, dataset):
        from trl import SFTTrainer, SFTConfig
        config = SFTConfig(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            learning_rate=self.cfg.lr,
            lr_scheduler_type=self.cfg.lr_scheduler,
            warmup_ratio=self.cfg.warmup_ratio,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            max_seq_length=self.cfg.max_seq_len,
            report_to="wandb",
        )
        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=self._tokenizer,
            args=config,
        )

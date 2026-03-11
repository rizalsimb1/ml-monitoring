"""Dataset formatting utilities for instruction tuning."""
from typing import Callable


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}

### Response:
{output}"""

CHATML_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def alpaca_format(example: dict) -> dict:
    """Format dataset sample in Alpaca instruction tuning format."""
    text = ALPACA_TEMPLATE.format(
        instruction=example.get("instruction", ""),
        output=example.get("output", ""),
    )
    return {"text": text}


def chatml_format(example: dict) -> dict:
    """Format dataset sample in ChatML format (OpenHermes, etc.)."""
    if "conversations" in example:
        parts = []
        for msg in example["conversations"]:
            role = "user" if msg["from"] == "human" else "assistant"
            parts.append(f"<|im_start|>{role}\n{msg['value']}<|im_end|>")
        return {"text": "\n".join(parts)}
    return chatml_format({"text": example.get("text", "")})


def estimate_vram(model_params_b: float, use_4bit: bool, batch_size: int = 4) -> dict:
    """Rough VRAM estimate for training configuration."""
    bytes_per_param = 0.5 if use_4bit else 2.0  # bf16
    model_gb = model_params_b * bytes_per_param
    overhead_gb = 4 + batch_size * 0.5  # activations, optimizer states
    total_gb = model_gb + overhead_gb
    return {
        "model_gb": round(model_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "total_gb": round(total_gb, 1),
        "recommended_gpu": "RTX 4090 (24GB)" if total_gb < 24 else "A100 (80GB)",
    }

# ml-monitoring

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/rizalsimb1/ml-monitoring?style=social)
![Issues](https://img.shields.io/github/issues/rizalsimb1/ml-monitoring)

> Fine-tune large language models (Llama 3, Mistral, Phi-3) with LoRA and QLoRA using HuggingFace PEFT. Includes dataset preparation, training, evaluation, and model merging.

## ✨ Features

- ✅ LoRA and QLoRA (4-bit quantization) support via HuggingFace PEFT
- ✅ Compatible with Llama 3, Mistral 7B, Phi-3, Qwen-2.5
- ✅ Instruction tuning with custom dataset formats (Alpaca, ShareGPT, ChatML)
- ✅ Flash Attention 2 integration for 3x faster training
- ✅ W&B and TensorBoard training metric logging
- ✅ Gradient checkpointing for memory efficiency
- ✅ Automatic merged model saving (LoRA weights merged into base)
- ✅ VRAM estimator — recommends optimal config for your GPU

## 🛠️ Tech Stack

`Python 3.11+` • `PyTorch 2.x` • `HuggingFace Transformers` • `PEFT` • `TRL` • `bitsandbytes` • `Datasets`

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rizalsimb1/ml-monitoring.git
cd ml-monitoring

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from trainer import LoRATrainer

trainer = LoRATrainer(
    base_model="meta-llama/Meta-Llama-3-8B",
    dataset="my-instruct-dataset.jsonl",
    output_dir="./lora-output",
    lora_rank=16,
    use_4bit=True,         # QLoRA
    epochs=3,
    lr=2e-4,
)

trainer.train()
trainer.merge_and_save("./final-model")
print("Fine-tuning complete!")

```

## 📁 Project Structure

```
ml-monitoring/
├── src/
│   └── main files
├── tests/
│   └── test files
├── requirements.txt
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/rizalsimb1">rizalsimb1</a></p>


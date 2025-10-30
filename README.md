# 🏟️ Lora-Evolution-Bench

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/🤗-PEFT-orange.svg)](https://huggingface.co/docs/peft)

**A comprehensive benchmarking suite for state-of-the-art parameter-efficient fine-tuning methods**

[Documentation](#documentation) • [Quick Start](#quick-start) • [Benchmarks](#benchmarks) • [Paper](#citation)

</div>

---

## 🎯 Overview

**Efficient Fine-Tuning Arena** is a modern implementation and benchmarking framework for comparing cutting-edge parameter-efficient fine-tuning (PEFT) techniques. This repository extends beyond traditional QLoRA to include the latest advancements in efficient LLM adaptation.

### Why This Repo?

- **🆚 Head-to-Head Comparisons**: Benchmark QLoRA, DoRA, ReFT, and more on identical datasets
- **💾 Memory Efficient**: Train 70B models on consumer GPUs with 24GB VRAM
- **📊 Comprehensive Metrics**: Track accuracy, memory usage, training time, and parameter counts
- **🔧 Production Ready**: Modular architecture for easy integration into your projects
- **📚 Educational**: Detailed tutorials and implementation guides for each method

---

## ✨ Supported Methods

| Method | Parameters | Memory | Speed | Best For |
|--------|-----------|---------|--------|----------|
| **QLoRA** | ~0.1-1% | ⭐⭐⭐⭐ | ⭐⭐⭐ | Baseline, proven reliability |
| **DoRA** | ~0.11-1% | ⭐⭐⭐⭐ | ⭐⭐⭐ | Better accuracy than LoRA |
| **ReFT/LoReFT** | ~0.01-0.05% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Ultra-efficient, fast training |
| **LoftQ Init** | Same as LoRA | ⭐⭐⭐⭐ | ⭐⭐⭐ | Better quantized initialization |
| **rsLoRA** | Same as LoRA | ⭐⭐⭐⭐ | ⭐⭐⭐ | Stable high-rank adaptation |

### Method Highlights

#### QLoRA (Baseline)
- 4-bit quantization with NormalFloat (NF4)
- Double quantization for extra memory savings
- Paged optimizers for handling memory spikes

#### DoRA (Weight-Decomposed LoRA)
- Decomposes weights into magnitude and direction
- **15-20% better accuracy** than standard LoRA on reasoning tasks
- Robust to rank selection with only 0.01% more parameters

#### ReFT (Representation Fine-Tuning)
- **10-50x fewer parameters** than LoRA
- Modifies hidden representations instead of weights
- Can achieve strong results with just 1,000 training examples

#### LoftQ Initialization
- Optimizes initialization for quantized models
- Significantly improves QLoRA convergence
- Minimal computational overhead

---

## 🚀 Quick Start

### Installation
Clone the repository
```python
clone https://github.com/yourusername/efficient-finetuning-arena.git
cd efficient-finetuning-arena
```
Create virtual environment
```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
Install dependencies
```
pip install -r requirements.txt
```



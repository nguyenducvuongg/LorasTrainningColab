#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🚀 Colab LoRA Studio - Automated One-Click Environment Setup"
echo "=========================================================="

# 1. Update pip and basic build tools
echo "📦 Installing core Python packaging tools..."
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch & ecosystem
echo "⚡ Installing PyTorch, Accelerate, Diffusers & Transformers..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers diffusers peft safetensors huggingface-hub bitsandbytes

# 3. Install Vision, Tagging, and Utilities
echo "🖼️ Installing Image Processing, ONNX, and Vision SDKs..."
pip install pillow opencv-python-headless einops numpy pyyaml toml tqdm requests rich google-genai openai

# 4. Clone training backends if not present
mkdir -p /content/backends
if [ ! -d "/content/backends/sd-scripts" ]; then
    echo "📥 Cloning Kohya sd-scripts..."
    git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /content/backends/sd-scripts
    pip install -r /content/backends/sd-scripts/requirements.txt || true
fi

if [ ! -d "/content/backends/ai-toolkit" ]; then
    echo "📥 Cloning AI-Toolkit for Flux.1..."
    git clone --depth 1 https://github.com/ostris/ai-toolkit.git /content/backends/ai-toolkit
    pip install -r /content/backends/ai-toolkit/requirements.txt || true
fi

# 5. Install local package in editable mode
echo "⚙️ Installing Colab LoRA Studio package..."
pip install -e .

echo "=========================================================="
echo "✅ Environment setup completed successfully!"
echo "=========================================================="

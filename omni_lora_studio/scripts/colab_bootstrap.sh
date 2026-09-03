#!/bin/bash
set -e

echo "🚀 [OmniLoRA Studio] Bắt đầu khởi tạo môi trường Colab siêu tốc..."

# 1. Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ Không tìm thấy GPU NVIDIA, đang chạy ở chế độ CPU."
fi

# 2. Cài đặt các gói cốt lõi và thư viện huấn luyện sd-scripts
echo "📦 Cài đặt thư viện phụ trợ & Kohya sd-scripts dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    "transformers>=4.40.0,<=4.48.3" \
    accelerate>=0.28.0 \
    timm>=0.9.16 \
    pydantic>=2.5.0 \
    pyyaml>=6.0 \
    rich>=13.7.0 \
    click>=8.1.0 \
    tqdm>=4.66.0 \
    pillow>=10.0.0 \
    psutil>=5.9.0 \
    prodigyopt>=1.0.0 \
    insightface>=0.7.3 \
    onnxruntime-gpu>=1.17.0 \
    opencv-python-headless>=4.9.0 \
    voluptuous>=0.13.1 \
    toml>=0.10.2 \
    einops>=0.7.0 \
    albumentations>=1.4.0 \
    open-clip-torch>=2.24.0 \
    lycoris-lora>=2.2.0 \
    dadaptation>=3.1 \
    protobuf>=3.20.0 \
    sentencepiece>=0.2.0

# 3. Cài đặt repo sd-scripts nếu chưa tồn tại
if [ ! -d "sd-scripts" ]; then
    echo "📥 Đang clone sd-scripts (Kohya ss)..."
    git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git
    if [ -f "sd-scripts/requirements.txt" ]; then
        pip install --quiet -r sd-scripts/requirements.txt || true
    fi
fi

# 4. Cài đặt package omni_lora
pip install --quiet -e .

echo "✅ [OmniLoRA Studio] Môi trường đã sẵn sàng 100%!"

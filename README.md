# 🎨 Colab LoRA Studio (All-in-One LoRA Training Suite)

> **Bộ công cụ toàn diện và chuyên nghiệp để huấn luyện LoRA cho mọi mô hình Diffusion (Flux.1, Flux-Kontext, Krea2-Raw, SDXL, Pony v6, SD 3.5, SD 1.5) tối ưu hóa riêng cho Google Colab (T4, L4, A100).**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuong/LorasTranning/blob/main/Colab_LoRA_Studio.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🌟 Tính Năng Nổi Bật (Key Features)

1. **Tự Động Nhận Diện Phần Cứng & Tối Ưu VRAM/RAM (Auto Hardware Profiler)**:
   * Tự động phát hiện GPU Google Colab (**Tesla T4, Nvidia L4, A100 40/80GB, V100**).
   * Tự động gán cấu hình tối ưu: FP8/NF4 Base model, 8-bit AdamW / Prodigy (D-Adaptation), Batch size, và Gradient Checkpointing thích ứng.
2. **Lưu Trữ 100% Trực Tiếp Vào Google Drive (Không Lo Mất Dữ Liệu)**:
   * Khởi tạo cây thư mục chuẩn hóa tại `MyDrive/Colab_LoRA_Studio/` lần đầu tiên.
   * **Smart Scan**: Quét thông minh ở các lần chạy tiếp theo để tuyệt đối không ghi đè dữ liệu cũ.
   * **Smart Downloader**: Tải Base Models, VAE, Text Encoders trực tiếp vào Google Drive một lần duy nhất; tự động kiểm tra và bỏ qua nếu đã tải, chỉ tải bù file còn thiếu.
3. **Hỗ Trợ Đa Dạng Bài Toán Huấn Luyện (Specialized Dataset Tasks)**:
   * **Character (Face & Body)**: Phân bổ trọng số `10_face`, `08_half_body`, `05_full_body`, `03_variations`.
   * **Art Style**: Phong cách nghệ thuật, hội hoạ, linework.
   * **Skin Texture & Detail Enhancement**: Tái tạo chi tiết da, lỗ chân lông, ánh sáng thực.
   * **Control & Upscale**: Huấn luyện Control-LoRA, adapter upscale, inpainting.
4. **Auto-Captioning Đa Nguồn & Chuẩn Hóa Tên File**:
   * **Google Gemini 1.5/2.0 API**: Tận dụng Cloud AI sinh mô tả cực nét mà không tốn VRAM Colab.
   * **DeepSeek / OpenAI Vision API**: Hỗ trợ qua endpoint tương thích.
   * **WD14 Tagger v3 (SmilingWolf)**: Gán Danbooru tags chuyên dụng cho Anime, 2D, Pony, Illustrious.
   * **JoyCaption Alpha / Florence-2**: Local VLM cho phong cách tả thực.
   * **Dataset Normalizer**: Tự động chuẩn hóa ảnh và đổi tên đồng bộ dạng `{prefix}_{index:04d}.png` và `.txt`.
5. **Master Notebook 1-Click (`Colab_LoRA_Studio.ipynb`)**:
   * Thiết kế tối ưu cho chế độ **"Run All" (Chạy tất cả)**, tích hợp giao diện Colab Forms trực quan.
   * Tự động khôi phục (**Auto-Resume**) nếu phiên làm việc trước bị ngắt kết nối.
   * Gửi ảnh Sample Preview & chỉ số Loss về điện thoại qua **Discord / Telegram Webhook**.

---

## 📁 Cấu Trúc Mã Nguồn (Repository Layout)

```
LorasTranning/
├── Colab_LoRA_Studio.ipynb                 # ★ MASTER ALL-IN-ONE NOTEBOOK ★
├── configs/                                # Cấu hình YAML/TOML mẫu cho từng loại GPU, Task & Model
│   ├── hardware/                           # colab_t4_free.yaml, colab_l4_pro.yaml, colab_a100_pro.yaml
│   ├── models/                             # flux_dev.yaml, flux_kontext.yaml, krea2_raw.yaml, sdxl_pony.toml...
│   └── tasks/                              # character_face_body.yaml, art_style.yaml, skin_texture_enhancement.yaml...
│
├── src/
│   └── lora_colab/                         # Core Python Package
│       ├── core/                           # hardware.py (Auto-Profiler), config.py, logger.py
│       ├── storage/                        # gdrive_manager.py, model_downloader.py, resume_manager.py
│       ├── dataset/                        # normalizer.py, cleaner.py, bucketing.py
│       │   └── captioning/                 # gemini_api.py, deepseek_api.py, wd14.py, joycaption.py
│       ├── engines/                        # aitoolkit_trainer.py, kohya_trainer.py, diffusers_trainer.py
│       ├── monitoring/                     # webhook.py, sample_generator.py
│       └── export/                         # converter.py, merger.py, uploader.py
│
├── scripts/
│   ├── colab_setup.sh                      # Cài đặt môi trường 1 lệnh
│   └── test_suite.py                       # Kiểm tra hệ thống
│
├── tests/                                  # Unit tests
├── pyproject.toml                          # Python package specification
└── requirements.txt                        # Dependencies list
```

---

## 🚀 Hướng Dẫn Sử Dụng Trên Google Colab

### Cách 1: Sử dụng Master Notebook (Khuyên dùng)
1. Đẩy mã nguồn này lên tài khoản GitHub của bạn (ví dụ: `https://github.com/your-username/LorasTranning.git`).
2. Mở file [`Colab_LoRA_Studio.ipynb`](file:///Users/nguyenducvuong/code/LorasTranning/Colab_LoRA_Studio.ipynb) trên Google Colab.
3. Bấm **Runtime -> Run all** (hoặc chạy tuần tự từng Cell).
4. Điền các tham số cần thiết (như Trigger Word, Model nền, API key nếu có) và hệ thống sẽ tự động thực hiện toàn bộ quá trình.

### Cách 2: Sử dụng dòng lệnh CLI trên Terminal / Colab
```bash
# 1. Khởi tạo thư mục Google Drive
lora-colab init-drive

# 2. Tải model nền tảng về Google Drive
lora-colab download-model --model flux-dev

# 3. Chuẩn hóa & Đổi tên ảnh dataset
lora-colab normalize-dataset --input-dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_character/10_face --prefix char_face

# 4. Gán nhãn tự động với Gemini API
export GEMINI_API_KEY="your-gemini-api-key"
lora-colab caption --dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_character/10_face --engine gemini --trigger "sks person"

# 5. Khởi chạy huấn luyện với tự động Resume
lora-colab train --config configs/models/flux_dev.yaml --resume
```

---

## ⚡ Bảng Tối Ưu Hóa Phần Cứng

| GPU | VRAM | Base Precision | Optimizer | Batch Size | Thời Gian Ước Tính (10 Epochs Flux) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nvidia A100** | 40/80 GB | BF16 Full | AdamW / Prodigy | 4 - 8 | ~15 - 25 phút |
| **Nvidia L4** | 24 GB | BF16 Native | Prodigy (Auto LR) | 2 - 4 | ~45 - 60 phút |
| **Tesla T4** | 16 GB | FP8 / NF4 | 8-bit AdamW | 1 (GradAccum 2-4) | ~2 - 3 giờ |

---

## 📄 License
Dự án được phân phối dưới giấy phép **Apache License 2.0**.

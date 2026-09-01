# 🎨 Colab LoRA Studio (All-in-One LoRA Training Suite)

<p align="center">
  <a href="https://colab.research.google.com/github/nguyenducvuongg/LorasTrainning/blob/main/Colab_LoRA_Studio.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="220">
  </a>
</p>

<p align="center">
  <b>Bộ công cụ toàn diện và chuyên nghiệp để huấn luyện LoRA cho mọi kiến trúc Diffusion trên Google Colab & Cloud GPU</b><br>
  <i>Tối ưu hóa đặc biệt cho Google Colab Pro (GPU L4 24GB, A100 40/80GB) và Colab Free (Tesla T4 16GB)</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg?style=flat-square" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/Models-Flux%20%7C%20SDXL%20%7C%20Pony%20%7C%20SD3.5%20%7C%20SD1.5-purple.svg?style=flat-square" alt="Supported Models">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License">
</p>

---

## 📑 Mục Lục (Table of Contents)
- [1. Giới Thiệu (Overview)](#1-giới-thiệu-overview)
- [2. Nút Chạy Trực Tiếp Trên Google Colab](#2-nút-chạy-trực-tiếp-trên-google-colab)
- [3. Tính Năng Nổi Bật (Key Features)](#3-tính-năng-nổi-bật-key-features)
- [4. Mô Hình Nền Tảng Hỗ Trợ (Supported Base Models)](#4-mô-hình-nền-tảng-hỗ-trợ-supported-base-models)
- [5. Hướng Dẫn Sử Dụng Chi Tiết Trên Colab (7 Bước Chạy 1-Click)](#5-hướng-dẫn-sử-dụng-chi-tiết-trên-colab-7-bước-chạy-1-click)
- [6. Cấu Trúc Dataset Theo Từng Bài Toán (Dataset Layout)](#6-cấu-trúc-dataset-theo-từng-bài-toán-dataset-layout)
- [7. Hệ Thống Gán Nhãn Tự Động (Auto-Captioning System)](#7-hệ-thống-gán-nhãn-tự-động-auto-captioning-system)
- [8. Bảng Tối Ưu Hóa Phần Cứng (Hardware Optimizer Matrix)](#8-bảng-tối-ưu-hóa-phần-cứng-hardware-optimizer-matrix)
- [9. Cấu Trúc Thư Mục Dự Án (Repository Structure)](#9-cấu-trúc-thư-mục-dự-án-repository-structure)
- [10. Hướng Dẫn Chạy Bằng Dòng Lệnh CLI](#10-hướng-dẫn-chạy-bằng-dòng-lệnh-cli)
- [11. Xử Lý Sự Cố Thường Gặp (Troubleshooting & FAQ)](#11-xử-lý-sự-cố-thường-gặp-troubleshooting--faq)

---

## 1. Giới Thiệu (Overview)

**Colab LoRA Studio** là giải pháp mã nguồn mở hoàn chỉnh giúp bạn huấn luyện mô hình LoRA (Low-Rank Adaptation) chất lượng cao nhất một cách tự động, an toàn và dễ dàng trên Google Colab.

Hệ thống giải quyết triệt để 4 vấn đề lớn nhất khi train LoRA trên Colab:
1. **Tránh mất dữ liệu**: Tải Base Models, lưu Checkpoint và xuất LoRA **100% trực tiếp vào Google Drive**.
2. **Tránh quá tải VRAM (OOM)**: Tự động đo đạc GPU (**L4, A100, T4, V100**) để áp dụng các kỹ thuật tối ưu bộ nhớ tiên tiến (**FP8/NF4 Base, 8-bit AdamW, Prodigy Adaptive LR, Disk Latent Caching**).
3. **Tiết kiệm thời gian chuẩn bị dữ liệu**: Tự động đổi tên ảnh chuẩn hóa `{prefix}_{index:04d}` và gán nhãn tự động qua **Gemini Vision API, DeepSeek, WD14 Tagger, JoyCaption**.
4. **Chống ngắt kết nối giữa chừng (Anti-Disconnect)**: Cơ chế **Auto-Resume** tự động tìm checkpoint dở dang trên Google Drive để tiếp tục phiên huấn luyện.

---

## 2. Nút Chạy Trực Tiếp Trên Google Colab

Nhấn vào huy hiệu dưới đây để mở trực tiếp Master Notebook trên Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/LorasTrainning/blob/main/Colab_LoRA_Studio.ipynb)

> **Link trực tiếp:** `https://colab.research.google.com/github/nguyenducvuongg/LorasTrainning/blob/main/Colab_LoRA_Studio.ipynb`

---

## 3. Tính Năng Nổi Bật (Key Features)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   COLAB LORA STUDIO                                    │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ ⚡ Auto Hardware Tuning   │ 💾 Smart Drive Workspace │ 🖼️ Multi-Engine Auto-Captioning   │
│ • Tự động nhận diện GPU  │ • Lưu 100% vào GDrive    │ • Google Gemini 1.5/2.0 API      │
│ • Profile cho T4/L4/A100 │ • Chống ghi đè dữ liệu   │ • DeepSeek / OpenAI Vision API   │
│ • FP8, BF16, 8bit-AdamW  │ • Tải model 1 lần dùng   │ • WD14 Tagger v3 (Danbooru Tags) │
│ • Prodigy (Auto LearnRate)│ • Auto-Resume checkpoint │ • JoyCaption / Florence-2 (VLM)  │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 4. Mô Hình Nền Tảng Hỗ Trợ (Supported Base Models)

| Nhóm Mô Hình | Các Model Hỗ Trợ | Framework Huấn Luyện | Định Dạng Xuất |
| :--- | :--- | :--- | :--- |
| **Flux Ecosystem** | **Flux.1-dev**, **Flux.1-schnell**, **Flux-Kontext** | **AI-Toolkit (ostris)** / **Kohya** | `.safetensors` (ComfyUI / WebUI Forge) |
| **SDXL & Anime** | **SDXL 1.0 Base**, **Pony Diffusion V6**, **Illustrious-XL**, **Animagine XL** | **Kohya_ss (sd-scripts)** | `.safetensors` (A1111 / ComfyUI / Forge) |
| **Next-Gen & Creative** | **Krea2-raw**, **SD 3.5 Medium**, **SD 3.5 Large**, **Z-Image**, **Sana** | **AI-Toolkit** / **Diffusers** | `.safetensors` |
| **SD 1.5 Classic** | **SD 1.5 Base**, **Realistic Vision v6.0**, **DreamShaper 8** | **Kohya_ss** / **Diffusers** | `.safetensors` |

---

## 5. Hướng Dẫn Sử Dụng Chi Tiết Trên Colab (7 Bước Chạy 1-Click)

Bạn chỉ cần mở notebook [`Colab_LoRA_Studio.ipynb`](https://colab.research.google.com/github/nguyenducvuongg/LorasTrainning/blob/main/Colab_LoRA_Studio.ipynb) và chọn **Runtime -> Run all** (hoặc chạy lần lượt 7 Cell):

### 🚀 Cell 1: Khởi Tạo Môi Trường & Smart Google Drive Setup
* Tự động clone mã nguồn mới nhất từ GitHub.
* Mount Google Drive và khởi tạo cây thư mục chuẩn tại `/content/drive/MyDrive/Colab_LoRA_Studio/` (Quét chống ghi đè dữ liệu cũ).
* Cài đặt trọn gói dependencies (PyTorch, Accelerate, Diffusers, Transformers, Bitsandbytes, AI-Toolkit, Kohya).

### ⚡ Cell 2: Nhận Diện Phần Cứng & Tự Động Tối Ưu (Auto Hardware Profile)
* Quét GPU hệ thống (**L4, A100, T4, V100**).
* Tự động cấu hình Mixed Precision (BF16/FP8), Batch size, Optimizer và Gradient Checkpointing thích hợp nhất.

### 📥 Cell 3: Smart Model Downloader (Lưu Trực Tiếp Vào Drive)
* Chọn Model nền cần tải (`flux-dev`, `flux-kontext`, `krea2-raw`, `sdxl-base`, `pony-v6`, `sd35-medium`, `sd15-base`).
* Quét Google Drive: **Nếu đã có sẵn -> Bỏ qua tải (tiết kiệm thời gian)**; **Nếu chưa có -> Tải trực tiếp vào Drive từ Public Hub**.

### 🖼️ Cell 4: Chuẩn Bị Dữ Liệu & Auto-Captioning Pipeline
* Chọn thư mục dataset (Face, Body, Character, Style, Skin Texture, Control-LoRA).
* Tự động chuẩn hóa ảnh RGB và đổi tên đồng bộ dạng `{prefix}_{index:04d}.png`.
* Chọn Captioner (Gemini Vision API, DeepSeek, WD14 Tagger, JoyCaption) và nhập Trigger Word (ví dụ: `sks person`).

### ⚙️ Cell 5: Cấu Hình Huấn Luyện (Training Configuration)
* Nhập tham số: Epochs (10-15), Network Rank (32-64), Alpha (16-32), Learning Rate.
* Nhập Discord Webhook URL hoặc Telegram Bot để nhận ảnh Preview trực tiếp về điện thoại.

### 🎯 Cell 6: Bắt Đầu Huấn Luyện (Training & Auto-Resume)
* Tự động phát hiện Checkpoint cũ trong Google Drive để kích hoạt **Auto-Resume** nếu phiên trước bị ngắt kết nối.
* Tiến hành huấn luyện, ghi log Loss và định kỳ lưu Checkpoint vào Google Drive.

### 📦 Cell 7: Kiểm Thử (Inference), Gộp LoRA & Upload
* Test Prompt sinh ảnh thử nghiệm với LoRA vừa train ngay trong Colab.
* Gộp (Merge) LoRA vào model gốc hoặc Upload trực tiếp lên **Hugging Face Hub / Civitai**.

---

## 6. Cấu Trúc Dataset Theo Từng Bài Toán (Dataset Layout)

Hệ thống đã chuẩn bị sẵn các thư mục phân bổ trọng số (Repeats) tối ưu cho từng mục đích:

```
MyDrive/Colab_LoRA_Studio/datasets/
├── 01_character/                           # Huấn luyện Nhân vật & Gương mặt
│   ├── 10_face/                            # 10-20 ảnh cận cảnh khuôn mặt (đa dạng góc nhìn & biểu cảm)
│   ├── 08_half_body/                       # 8-15 ảnh nửa người (chi tiết trang phục)
│   ├── 05_full_body/                       # 5-10 ảnh toàn thân (dáng đứng, chuyển động)
│   └── 03_variations/                      # 3-5 ảnh hoạt cảnh phức tạp
│
├── 02_style/                               # Huấn luyện Phong cách nghệ thuật (Art Style)
│   └── 10_style_art/                       # 20-50 ảnh đa dạng chủ đề nhưng cùng nét vẽ
│
├── 03_enhancement/                         # Huấn luyện Tái tạo da, Siêu chi tiết & Upscale
│   ├── condition/                          # Ảnh đầu vào (độ phân giải thường, mờ, raw)
│   └── target/                             # Ảnh đích siêu nét, chi tiết da, ánh sáng thực
│
└── 04_control/                             # Huấn luyện Control-LoRA
    ├── conditioning_images/                # Canny / Depth / Pose / Mask
    └── ground_truth_images/                # Ảnh kết quả tương ứng
```

---

## 7. Hệ Thống Gán Nhãn Tự Động (Auto-Captioning System)

| Công Cụ Gán Nhãn | Thể Loại Phù Hợp | Ưu Điểm | Cách Sử Dụng |
| :--- | :--- | :--- | :--- |
| **Google Gemini 1.5/2.0 API** | Chân dung thực tế, Phong cách, Skin texture, Flux | Mô tả tự nhiên cực nét, hiểu sâu ngữ cảnh, **0% tiêu tốn VRAM Colab** | Nhập `GEMINI_API_KEY` |
| **DeepSeek / OpenAI Vision** | Chân dung, Tả thực, Chi tiết | Rất chi tiết về ánh sáng và bố cục | Nhập API Key & Base URL |
| **SmilingWolf WD14 Tagger v3** | Anime, Manga, 2D Art, Pony v6, Illustrious | Trích xuất Danbooru tags chuẩn xác, hỗ trợ lọc Blacklist tags | Chạy Local trên Colab |
| **JoyCaption / Florence-2** | Photorealism, Phong cảnh, Đồ vật | Mô tả văn phong tự nhiên phong phú | Chạy Local VLM |

---

## 8. Bảng Tối Ưu Hóa Phần Cứng (Hardware Optimizer Matrix)

| GPU | VRAM | Base Precision | Optimizer | Batch Size | Thời Gian Ước Tính (10 Epochs Flux) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nvidia A100** (Colab Pro+) | 40 / 80 GB | BF16 Full | AdamW / Prodigy | 4 - 8 | ~15 - 25 phút |
| **Nvidia L4** (Colab Pro) | 24 GB | BF16 Native | Prodigy (Auto LR) | 2 - 4 | ~40 - 55 phút |
| **Tesla T4** (Colab Free) | 16 GB | FP8 / NF4 | 8-bit AdamW | 1 (GradAccum 2-4) | ~2 - 3 giờ |

---

## 9. Cấu Trúc Thư Mục Dự Án (Repository Structure)

```
LorasTrainning/
├── Colab_LoRA_Studio.ipynb                 # ★ Master Notebook tích hợp trọn gói ★
├── configs/                                # Các mẫu cấu hình YAML/TOML tối ưu sẵn
│   ├── hardware/                           # colab_t4_free.yaml, colab_l4_pro.yaml, colab_a100_pro.yaml
│   ├── models/                             # flux_dev.yaml, flux_kontext.yaml, krea2_raw.yaml, sdxl_pony.toml...
│   └── tasks/                              # character_face_body.yaml, art_style.yaml, skin_texture_enhancement.yaml...
│
├── src/
│   └── lora_colab/                         # Thư viện lõi Python
│       ├── core/                           # hardware.py (Auto-Profiler), config.py, logger.py
│       ├── storage/                        # gdrive_manager.py, model_downloader.py, resume_manager.py
│       ├── dataset/                        # normalizer.py, cleaner.py, bucketing.py
│       │   └── captioning/                 # gemini_api.py, deepseek_api.py, wd14.py, joycaption.py
│       ├── engines/                        # aitoolkit_trainer.py, kohya_trainer.py, diffusers_trainer.py
│       ├── monitoring/                     # webhook.py, sample_generator.py
│       └── export/                         # converter.py, merger.py, uploader.py
│
├── scripts/
│   ├── colab_setup.sh                      # Script cài đặt môi trường
│   ├── test_suite.py                       # Kịch bản kiểm tra toàn diện hệ thống
│   └── run_unit_tests.py                   # Bộ Unit Tests chuẩn Unittest
│
├── pyproject.toml                          # Chuẩn đóng gói Package Python
└── requirements.txt                        # Danh sách dependencies
```

---

## 10. Hướng Dẫn Chạy Bằng Dòng Lệnh CLI

Ngoài Notebook, bạn hoàn toàn có thể sử dụng công cụ CLI `lora-colab`:

```bash
# 1. Khởi tạo cây thư mục Google Drive
lora-colab init-drive

# 2. Tải model nền tảng về Google Drive
lora-colab download-model --model flux-dev

# 3. Chuẩn hóa tên ảnh và chuyển sang định dạng PNG
lora-colab normalize-dataset --input-dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_character/10_face --prefix char_face

# 4. Gán nhãn tự động với Gemini API
export GEMINI_API_KEY="your-gemini-api-key"
lora-colab caption --dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_character/10_face --engine gemini --trigger "sks person"

# 5. Khởi chạy huấn luyện với tính năng Auto-Resume
lora-colab train --config configs/models/flux_dev.yaml --resume
```

---

## 11. Xử Lý Sự Cố Thường Gặp (Troubleshooting & FAQ)

### ❓ Khi bị ngắt kết nối Colab giữa chừng (Session Disconnected) thì làm thế nào?
> **Trả lời:** Đừng lo lắng! Tất cả checkpoints định kỳ đã được lưu an toàn trực tiếp vào Google Drive (`outputs/checkpoints/`). Bạn chỉ cần mở lại notebook, bật tùy chọn `ENABLE_AUTO_RESUME = True` ở Cell 6 và chạy tiếp, hệ thống sẽ tự động tìm checkpoint gần nhất và train nối tiếp.

### ❓ Làm sao để nhận ảnh Preview và thông báo khi đang train?
> **Trả lời:** Ở Cell 5, hãy dán link `DISCORD_WEBHOOK_URL` của kênh Discord của bạn (hoặc Telegram Bot Token). Mỗi khi tạo xong ảnh Sample hoặc kết thúc Epoch, bot sẽ tự động gửi ảnh mẫu và chỉ số Loss về điện thoại của bạn.

### ❓ Train Flux.1 trên Colab Free GPU T4 có bị tràn RAM/VRAM không?
> **Trả lời:** Không. Hệ thống đã tự động tích hợp **FP8 Base Quantization**, **T5-XXL 4-bit**, **Disk Latent Cache** và **Fused Backward Pass**, giúp quá trình train Flux.1 LoRA chỉ chiếm ~11.5 - 13.5 GB VRAM (hoàn toàn vừa vặn trong 16GB của T4).

---

## 📄 Bản Quyền & Giấy Phép (License)

Dự án được phát triển và phân phối dưới giấy phép **Apache License 2.0**.
Mọi đóng góp (Pull Request, Issue) đều được chào đón!

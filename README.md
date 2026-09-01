# 🎨 Colab LoRA Studio (All-in-One LoRA Training Suite)

<p align="center">
  <a href="https://colab.research.google.com/github/nguyenducvuongg/LorasTrainningColab/blob/main/Colab_LoRA_Studio.ipynb">
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
- [1. Nút Chạy Trực Tiếp Trên Google Colab](#1-nút-chạy-trực-tiếp-trên-google-colab)
- [2. Cấu Trúc Phân Loại Dữ Liệu Trainning (Dataset Categories)](#2-cấu-trúc-phân-loại-dữ-liệu-trainning-dataset-categories)
- [3. Hướng Dẫn Chi Tiết Các Thông Số Huấn Luyện (Hyperparameters Guide)](#3-hướng-dẫn-chi-tiết-các-thông-số-huấn-luyện-hyperparameters-guide)
- [4. Mô Hình Nền Tảng Hỗ Trợ (Supported Base Models)](#4-mô-hình-nền-tảng-hỗ-trợ-supported-base-models)
- [5. Hướng Dẫn 7 Bước Chạy 1-Click Trên Google Colab](#5-hướng-dẫn-7-bước-chạy-1-click-trên-google-colab)
- [6. Hệ Thống Gán Nhãn Tự Động (Auto-Captioning System)](#6-hệ-thống-gán-nhãn-tự-động-auto-captioning-system)
- [7. Bảng Tối Ưu Hóa Theo Từng Loại GPU (Hardware Matrix)](#7-bảng-tối-ưu-hóa-theo-từng-loại-gpu-hardware-matrix)
- [8. Hướng Dẫn Chạy Bằng Dòng Lệnh CLI](#8-hướng-dẫn-chạy-bằng-dòng-lệnh-cli)
- [9. Xử Lý Sự Cố Thường Gặp (FAQ & Troubleshooting)](#9-xử-lý-sự-cố-thường-gặp-faq--troubleshooting)

---

## 1. Nút Chạy Trực Tiếp Trên Google Colab

Nhấn vào huy hiệu dưới đây để mở trực tiếp Master Notebook trên Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/LorasTrainningColab/blob/main/Colab_LoRA_Studio.ipynb)

> **Link trực tiếp:** `https://colab.research.google.com/github/nguyenducvuongg/LorasTrainningColab/blob/main/Colab_LoRA_Studio.ipynb`

---

## 2. Cấu Trúc Phân Loại Dữ Liệu Trainning (Dataset Categories)

Hệ thống được thiết lập phân chia rõ ràng theo từng mục đích huấn luyện (lưu trực tiếp tại `MyDrive/Colab_LoRA_Studio/datasets/` và **không giới hạn số lượng hay loại file**):

```
Google Drive: MyDrive/Colab_LoRA_Studio/datasets/
├── 01_face/                                # 👤 Train chuyên sâu KHUÔN MẶT
│   └── (10-30 ảnh cận cảnh mặt, đa dạng góc nhìn 45°, 90°, ngước lên, cúi xuống, biểu cảm cười, nghiêm túc)
│
├── 02_character/                           # 🧍 Train TOÀN DIỆN NHÂN VẬT (Mặt + Nửa người + Toàn thân + Body + Trang phục)
│   └── (15-50 ảnh bao gồm: cận mặt, nửa thân trên, toàn thân đứng/ngồi, trang phục đặc trưng)
│
├── 03_style/                               # 🎨 Train PHONG CÁCH NGHỆ THUẬT (Art Style, Nét vẽ, Hội hoạ)
│   └── (20-60 ảnh phong cách đồng nhất với nhiều chủ thể khác nhau: người, phong cảnh, đồ vật)
│
├── 04_skin_enhancement/                    # 🔬 Train CHI TIẾT DA, LỖ CHÂN LÔNG & UPSCALE
│   └── (10-40 ảnh siêu nét về kết cấu da thực tế, dynamic range, ánh sáng studio thực)
│
├── 05_control/                             # 🕹️ Train CONTROL-LORA / ẢNH ĐIỀU KIỆN
│   └── (Các cặp ảnh điều kiện Canny, Depth, Pose, Inpainting)
│
├── 06_custom/                              # 📁 THƯ MỤC TÙY CHỈNH BẤT KỲ
│   └── (Bạn có thể đặt tên bất kỳ hoặc trỏ đường dẫn tùy ý trên Drive, không giới hạn file!)
│
└── raw_uploads/                            # Nơi bạn ném ảnh zip hoặc ảnh thô tải lên lần đầu
```

---

## 3. Hướng Dẫn Chi Tiết Các Thông Số Huấn Luyện (Hyperparameters Guide)

| Thông Số | Giá Trị Đề Xuất | Giải Thích & Kinh Nghiệm Tinh Chỉnh |
| :--- | :--- | :--- |
| **`Batch_Size`** | • **T4 (16GB)**: `1`<br>• **L4 (24GB)**: `2 - 4`<br>• **A100 (40/80GB)**: `4 - 8` | Số lượng ảnh mô hình xử lý cùng một lúc. Batch size càng lớn thì tốc độ train càng nhanh và gradient càng ổn định. |
| **`Optimizer`** | • **`Prodigy` (Khuyên dùng)**<br>• **`AdamW8bit`** (Tiết kiệm VRAM)<br>• **`AdamW`** (A100 full speed)<br>• **`Adafactor`** (Siêu nhẹ) | **Prodigy (D-Adaptation)** là trình tối ưu thông minh nhất hiện nay, tự động thích ứng và tính toán Learning Rate chuẩn xác nhất cho từng bước (tránh cháy model). |
| **`Learning_Rate`** | • **`1.0`** (Bắt buộc nếu dùng Prodigy)<br>• **`1e-4`** (`0.0001` - Chuẩn cho AdamW / Flux / SDXL)<br>• **`5e-5`** (`0.00005` - Tinh chỉnh nhẹ)<br>• **`2e-4`** (`0.0002` - Học nhanh)<br>• **`5e-4`** (`0.0005` - Mặc định cho SD 1.5) | Tốc độ học của mạng LoRA. Nếu dùng **Prodigy**, hãy luôn chọn **`1.0`** (vì thuật toán D-Adaptation sẽ tự điều chỉnh về ngưỡng tối ưu từ `1e-4` đến `1e-6`). |
| **`Epochs`** | • **Khuôn mặt / Face**: `10 - 15`<br>• **Nhân vật / Character**: `10 - 15`<br>• **Phong cách / Style**: `15 - 20`<br>• **Skin / Detailer**: `8 - 12` | Số chu kỳ mô hình quét qua toàn bộ tập dữ liệu ảnh của bạn. |
| **`Repeats`** | `10` (Mặc định) | Số lần lặp lại mỗi bức ảnh trong một Epoch. |
| **`Max_Train_Steps`** | `0` (Tự động tính theo Epochs) | Tổng số bước huấn luyện. Công thức tính: $$\text{Total Steps} = \left(\frac{\text{Số ảnh} \times \text{Repeats}}{\text{Batch Size}}\right) \times \text{Epochs}$$. Ngưỡng đẹp nhất cho LoRA nhân vật thường là **`1500 - 3000` steps**. |
| **`Network_Dim (Rank)`** | `32` hoặc `64` | Dung lượng bộ nhớ của LoRA. Rank càng cao (ví dụ: `64` hoặc `128`) mô hình càng học được nhiều chi tiết phức tạp. Rank `32` là tỷ lệ vàng giữa chất lượng và kích thước file. |
| **`Network_Alpha`** | `16` (hoặc bằng Rank) | Tỷ lệ tác động của LoRA. Quy tắc vàng: Đặt `Alpha = Rank / 2` (ví dụ: Rank 32 -> Alpha 16) hoặc `Alpha = Rank`. |
| **`Resolution`** | `1024` (Flux / SDXL) hoặc `512` (SD 1.5) | Độ phân giải mục tiêu. Hệ thống tự động bật **Aspect Ratio Bucketing** để học được cả ảnh dọc, ảnh ngang mà không bị méo hình. |

---

## 4. Mô Hình Nền Tảng Hỗ Trợ (Supported Base Models)

| Nhóm Mô Hình | Các Model Hỗ Trợ | Framework Lõi | Tối Ưu Hóa VRAM |
| :--- | :--- | :--- | :--- |
| **Flux Ecosystem** | **Flux.1-dev**, **Flux.1-schnell**, **Flux-Kontext** | **AI-Toolkit (ostris)** / **Kohya** | FP8 Base Model, NF4, T5 4-bit, Fused Backward Pass |
| **SDXL & Anime** | **SDXL 1.0 Base**, **Pony Diffusion V6**, **Illustrious-XL**, **Animagine XL** | **Kohya_ss (sd-scripts)** | BF16/FP16, Cache Latents to Disk, Prodigy |
| **Next-Gen & Creative** | **Krea2-raw**, **SD 3.5 Medium**, **SD 3.5 Large**, **Z-Image**, **Sana** | **AI-Toolkit** / **Diffusers** | FlashAttention-2, SDPA, LoRA / DoRA |
| **SD 1.5 Classic** | **SD 1.5 Base**, **Realistic Vision v6.0**, **DreamShaper 8** | **Kohya_ss** / **Diffusers** | FP16, Tốc độ huấn luyện cực nhanh |

---

## 5. Hướng Dẫn 7 Bước Chạy 1-Click Trên Google Colab

Mở file [`Colab_LoRA_Studio.ipynb`](https://colab.research.google.com/github/nguyenducvuongg/LorasTrainningColab/blob/main/Colab_LoRA_Studio.ipynb) và chọn **Runtime -> Run all**:

```
[Cell 1: 🚀 Khởi tạo Môi trường & Mount Google Drive]
  ├── Tự động Clone repo và cài đặt Dependencies
  └── Tạo cây thư mục chuẩn tại MyDrive/Colab_LoRA_Studio/ (Chống ghi đè)

[Cell 2: ⚡ Nhận diện Phần cứng GPU & Auto Tuning]
  └── Đo GPU (L4, A100, T4) và gán VRAM Optimizer Profile tự động

[Cell 3: 📥 Smart Model Downloader (Lưu trực tiếp vào Drive)]
  └── Chọn Model -> Quét Drive (Bỏ qua nếu đã có, chỉ tải bù file thiếu từ Hub)

[Cell 4: 🖼️ Chuẩn bị Dữ liệu & Auto-Captioning AI]
  ├── Chọn: 01_face, 02_character, 03_style, 04_skin_enhancement hoặc Custom
  ├── Tự động đổi tên đồng bộ {prefix}_{0001}.png
  └── Gán nhãn tự động bằng Gemini Vision API / DeepSeek / WD14 / JoyCaption

[Cell 5: ⚙️ Cấu hình Huấn luyện & Tinh chỉnh Tham số]
  └── Chọn Optimizer (Prodigy / AdamW8bit), Learning Rate (1e-4 / 1.0), Batch Size, Epochs

[Cell 6: 🎯 Bắt đầu Huấn luyện & Auto-Resume]
  ├── Quét tìm Checkpoint cũ trên Drive -> Tự động Resume nếu bị rớt kết nối
  └── Huấn luyện, định kỳ lưu Checkpoint vào Drive và gửi ảnh Preview qua Discord Webhook

[Cell 7: 📦 Kiểm thử (Inference), Gộp LoRA & Upload]
  └── Test Prompt sinh ảnh ngay trong Colab, Merge LoRA vào Checkpoint gốc hoặc Upload lên HuggingFace
```

---

## 6. Hệ Thống Gán Nhãn Tự Động (Auto-Captioning System)

1. **Google Gemini Vision API (1.5 Flash/Pro, 2.0 Flash)**: Khuyên dùng số 1! Mô tả văn phong tự nhiên cực chi tiết (khuôn mặt, góc chụp, ánh sáng, biểu cảm, chất liệu da), **0% tiêu tốn VRAM Colab**.
2. **DeepSeek / OpenAI Vision API**: Hỗ trợ qua endpoint tương thích OpenAI.
3. **SmilingWolf WD14 Tagger v3**: Trích xuất Danbooru tags hoàn hảo cho Anime, Manga, Pony Diffusion, Illustrious-XL.
4. **JoyCaption Alpha / Florence-2**: Local Vision Language Model cho ảnh tả thực (Photorealism).

---

## 7. Bảng Tối Ưu Hóa Theo Từng Loại GPU (Hardware Matrix)

| GPU Colab | VRAM | Base Precision | Optimizer Đề Xuất | Batch Size | Thời Gian (10 Epochs Flux) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nvidia A100** (Colab Pro+) | 40 / 80 GB | BF16 Full Precision | AdamW / Prodigy | 4 - 8 | ~15 - 25 phút |
| **Nvidia L4** (Colab Pro) | 24 GB | BF16 Native | Prodigy (Auto LR) | 2 - 4 | ~40 - 55 phút |
| **Tesla T4** (Colab Free) | 16 GB | FP8 / NF4 (Quantized) | 8-bit AdamW | 1 (GradAccum 2-4) | ~2 - 3 giờ |

---

## 8. Hướng Dẫn Chạy Bằng Dòng Lệnh CLI

```bash
# 1. Khởi tạo cấu trúc thư mục Google Drive
lora-colab init-drive

# 2. Tải model nền tảng về Google Drive
lora-colab download-model --model flux-dev

# 3. Chuẩn hóa tên ảnh trong thư mục face
lora-colab normalize-dataset --input-dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_face --prefix face

# 4. Gán nhãn tự động với Gemini API
export GEMINI_API_KEY="your-gemini-api-key"
lora-colab caption --dir /content/drive/MyDrive/Colab_LoRA_Studio/datasets/01_face --engine gemini --trigger "sks person"

# 5. Khởi chạy huấn luyện với tính năng Auto-Resume
lora-colab train --config configs/models/flux_dev.yaml --resume
```

---

## 9. Xử Lý Sự Cố Thường Gặp (FAQ & Troubleshooting)

* **Q: Bị ngắt kết nối Colab (Session Disconnect) có mất file không?**
  * **A:** Hoàn toàn không! Mọi checkpoint định kỳ và LoRA hoàn thiện đều được lưu trực tiếp tại Google Drive `outputs/checkpoints/`. Mở lại notebook, bật `ENABLE_AUTO_RESUME = True` ở Cell 6 là hệ thống tự động train tiếp từ bước trước.
* **Q: Làm sao nhận ảnh test preview về điện thoại trong lúc train?**
  * **A:** Dán URL Discord Webhook vào ô `DISCORD_WEBHOOK_URL` ở Cell 5. Cứ mỗi lần tạo ảnh sample hoặc xong Epoch, kết quả sẽ gửi thẳng về kênh Discord của bạn.
* **Q: Tôi có nhiều thư mục ảnh khác nhau thì xử lý thế nào?**
  * **A:** Chọn `06_custom_path` ở Cell 4 và dán đường dẫn thư mục bất kỳ trên Drive của bạn. Hệ thống hỗ trợ xử lý không giới hạn số lượng và phân cấp file ảnh.

---

## 📄 License
Dự án được phân phối theo giấy phép **Apache License 2.0**.

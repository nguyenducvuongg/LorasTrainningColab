# 🏗️ Kiến Trúc Hệ Thống OmniLoRA Studio

## 1. Mục Tiêu Thiết Kế
OmniLoRA Studio được xây dựng nhằm giải quyết triệt để 3 vấn đề lớn nhất của việc huấn luyện LoRA hiện nay:
1. **Độ tương đồng thấp hoặc dễ bị cháy (Likeness degradation / Overfitting)**: Khắc phục bằng bộ ba công nghệ:
   - **Identity Isolator**: Lọc bỏ đặc điểm khuôn mặt cố định, dồn 100% nhận diện vào trigger token.
   - **Multi-Scale Face Dataset**: Sinh tự động crop cận mặt 1024px + nửa thân + toàn thân.
   - **DoRA (Weight-Decomposed LoRA)**: Phân rã hướng & độ lớn trọng số, tương đương Full Fine-Tuning.
2. **Đa dạng kiến trúc mô hình (Multi-Model Fragmentation)**: Tự động điều phối giữa **Kohya-ss**, **AI-Toolkit** và **Musubi-Tuner** cho **Flux.1, SDXL, Pony V6, Illustrious-XL, SD 3.5, SD 1.5, Krea2, Z-Image (Kolors), Wan 2.1**.
3. **Giới hạn phần cứng trên Google Colab**: Tự động nhận diện GPU T4 (16GB), L4 (24GB), A100 (40/80GB), chuyển đổi FP8/NF4, Text-Encoder CPU offloading và Local NVMe SSD staging chống treo Google Drive FUSE.

## 2. Luồng Dữ Liệu Toàn Diện (End-to-End Dataflow)

```
[Raw Images / ZIP / Google Drive]
               │
               ▼
   [DatasetPreprocessor] ──> Khử EXIF xoay, chuyển RGB chuẩn
               │
               ▼
  [FaceAwareCropGenerator] ──> Tách 3 tỷ lệ: Face (1024px) | Medium | Full
               │
               ▼
     [CaptioningEngine] ─────> Florence-2 / JoyCaption / WD14
               │
               ▼
     [IdentityIsolator] ─────> Triệt tiêu miêu tả mặt cố định, khóa nhận diện vào [trigger]
               │
               ▼
    [AspectRatioBucketer] ───> Chia bucket độ phân giải từ 512px đến 1536px
               │
               ▼
      [EngineFactory] ───────> Auto-dispatch:
                                • Flux.1 ────────> KohyaFluxTrainer (FP8/Flow-Match)
                                • SDXL/Pony/SD3.5> KohyaSDXLTrainer (DoRA/Min-SNR)
                                • Krea2 ─────────> AIToolkitTrainer
                                • Z-Image/Wan2.1 > MusubiTrainer
                                • SD 1.5 ────────> KohyaSD15Trainer
               │
               ▼
   [LikenessMeter & ArcFace] ─> Đo Cosine Similarity % trực tiếp theo thời gian thực
               │
               ▼
  [BestCheckpointSelector] ──> Tự động lưu và đồng bộ BEST_100_LIKENESS_MODEL.safetensors về Google Drive
```

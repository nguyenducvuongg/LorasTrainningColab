# 🚀 Hướng Dẫn Vận Hành 1-Click Trên Google Colab

## 1. Bước 1: Mở Notebook Trên Colab
Chỉ cần nhấn vào huy hiệu **Open In Colab** tại `omni_lora_studio/README.md`.

## 2. Bước 2: Chuẩn Bị Dữ Liệu
Bạn chỉ cần nén các bức ảnh của mình thành file `dataset.zip` và tải lên Google Drive tại thư mục:
`MyDrive/OmniLoRA_Studio/dataset.zip` (hoặc ném ảnh trực tiếp vào một thư mục trên Drive).

## 3. Bước 3: Chạy Huấn Luyện 1-Click
Trong Master Notebook `OmniLoRA_Studio_Colab.ipynb`:
1. **Cell 1**: Khởi tạo môi trường & mount Google Drive (tự động nhận diện GPU T4 / L4 / A100).
2. **Cell 2**: Chọn mô hình (Flux.1, SDXL, Pony, SD3.5, Krea2, Z-Image) và Preset `100% Face Likeness`.
3. **Cell 3**: Bấm chạy! Toàn bộ tiến trình tiền xử lý, gán nhãn, bucketing và train sẽ tự động vận hành.
4. **Cell 4**: Xem trước kết quả ảnh sinh thử nghiệm kèm điểm số tương đồng `% Likeness` và nhận file `.safetensors` đã đồng bộ sẵn trong Google Drive!

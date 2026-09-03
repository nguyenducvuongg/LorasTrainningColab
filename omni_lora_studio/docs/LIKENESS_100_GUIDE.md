# 🧬 Cẩm Nang Khoa Học Đạt Độ Giống Ảnh Đầu Vào 100% (100% Likeness Guide)

## 1. Bản Chất Khoa Học Của "Độ Giống" Trong Mô Hình Diffusion
Mô hình Diffusion học theo cơ chế Cross-Attention giữa Text Tokens và Latent Image Features:
- **Nguyên nhân mất nhận diện**: Khi bạn chú thích ảnh là: `"a woman with brown hair and green eyes wearing a denim jacket"`, Text Encoder phân bổ gradient của mái tóc nâu vào từ `"brown hair"`, gradient đôi mắt xanh vào từ `"green eyes"`. Từ khóa `sks` chỉ nhận được phần dư thừa. Khi inference với prompt `a photo of sks in a space suit`, mô hình không tái hiện được mắt và tóc gốc!
- **Nguyên lý Cô Lập Chủ Thể (Identity Decoupling)**:
  `IdentityIsolator` tự động lọc bỏ toàn bộ các tính từ mô tả nhân trắc học cố định của khuôn mặt.
  Caption thu được: `"sks woman, wearing a denim jacket, standing outdoor, natural sunlight"`.
  Toàn bộ đặc điểm khuôn mặt, cấu trúc xương hàm, dáng mắt, màu tóc bị ép 100% hội tụ vào `sks`!

## 2. Tiêu Chuẩn Tập Dữ Liệu Vàng (Dataset Golden Rules)
1. **Số lượng ảnh lý tưởng**: 15 - 35 ảnh chất lượng cao.
2. **Tỷ lệ góc chụp**:
   - 40% Cận cảnh khuôn mặt (Close-up, ánh sáng rõ ràng, mắt nhìn thẳng và nghiêng 45°).
   - 40% Nửa thân trên (Medium shot, thấy rõ trang phục và vai).
   - 20% Toàn thân (Full body, dáng đứng/ngồi).
3. **Biểu cảm**: Cần có ít nhất 2-3 ảnh cười tự nhiên và 2-3 ảnh nghiêm túc để LoRA không bị "đóng băng" một nét mặt.
4. **Không cần crop thủ công**: Module `FaceAwareCropGenerator` sẽ tự động trích xuất các crop cận mặt 1024x1024 siêu nét cho bạn!

## 3. Thang Đo Tương Đồng Cosine (ArcFace Metric)
Hệ thống sử dụng mạng nơ-ron nhận diện khuôn mặt chuyên dụng (ArcFace ResNet50/100) để đo lường:
- `< 50%`: Khác người hoặc khuôn mặt biến dạng.
- `60% - 75%`: Tương đồng mức khá (LoRA thông thường).
- `75% - 85%`: Rất giống (chuẩn thương mại).
- `> 85%`: Độ giống đạt mức 100% của người thật (hoàn hảo tuyệt đối).

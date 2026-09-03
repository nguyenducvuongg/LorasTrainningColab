import re
from typing import List, Set

class IdentityIsolator:
    """
    BỘ LỌC CÔ LẬP CHỦ THỂ (CONCEPT & IDENTITY DECOUPLING FILTER).
    Đây là cốt lõi khoa học để đạt độ tương đồng 100% so với ảnh thật:
    - Loại bỏ các từ miêu tả đặc điểm khuôn mặt cố định (màu mắt, màu tóc, dáng mũi, xương gò má, v.v.).
    - Nhờ đó, MÔ HÌNH BỊ ÉP PHẢI GÁN TOÀN BỘ ĐẶC ĐIỂM KHUÔN MẶT CỐ ĐỊNH NÀY VÀO TỪ KÍCH HOẠT (TRIGGER TOKEN).
    - Giữ lại toàn bộ bối cảnh thay đổi: trang phục, ánh sáng, góc máy, phông nền, biểu cảm.
    """

    # Danh sách các đặc điểm nhận diện khuôn mặt cố định cần triệt tiêu khỏi caption
    INVARIANT_FACIAL_PATTERNS = [
        # Tóc
        r"\b(blonde|brunette|black|brown|red|ginger|curly|straight|wavy|short|long)\s+hair\b",
        r"\bhair\s+(is|styled in)\s+[a-zA-Z\s]+\b",
        # Mắt
        r"\b(blue|brown|hazel|green|dark|amber|grey)\s+eyes\b",
        r"\b(almond-shaped|wide|narrow)\s+eyes\b",
        # Mặt & Da
        r"\b(oval|round|square|heart-shaped)\s+face\b",
        r"\b(fair|pale|tan|dark|smooth|flawless)\s+skin\b",
        r"\b(high\s+cheekbones|sharp\s+jawline|pointed\s+chin)\b",
        r"\b(slender|straight|button)\s+nose\b",
        r"\b(thin|full|plump)\s+lips\b",
        # Chủng tộc / Nhận diện nhân khẩu học
        r"\b(caucasian|asian|latina|latino|hispanic|african|european)\b",
        # Tuổi tác cố định
        r"\b(in\s+her\s+early\s+\d+s|in\s+his\s+early\s+\d+s|young|middle-aged|elderly)\b",
    ]

    @classmethod
    def purify_caption(
        cls, 
        raw_caption: str, 
        trigger_word: str = "sks", 
        class_word: str = "person"
    ) -> str:
        """
        Lọc sạch các từ miêu tả nhận diện cố định và đưa trigger_word lên đầu câu.
        """
        if not raw_caption or not raw_caption.strip():
            return f"{trigger_word} {class_word}"

        caption = raw_caption.strip()

        # Loại bỏ các mẫu đặc điểm cố định
        for pattern in cls.INVARIANT_FACIAL_PATTERNS:
            caption = re.sub(pattern, "", caption, flags=re.IGNORECASE)

        # Chuẩn hóa khoảng trắng và dấu phẩy dư thừa
        caption = re.sub(r",\s*,+", ",", caption)
        caption = re.sub(r"\s+", " ", caption)
        caption = caption.strip(" ,.-")

        # Đảm bảo câu không bị rỗng sau khi lọc
        if not caption:
            return f"{trigger_word} {class_word}"

        # Đưa trigger_word và class_word lên vị trí đầu tiên để có attention weight cao nhất
        trigger_prefix = f"{trigger_word} {class_word}"
        if not caption.lower().startswith(trigger_prefix.lower()):
            caption = f"{trigger_prefix}, {caption}"

        return caption

import requests
from typing import Optional
from ..core.logger import setup_logger

logger = setup_logger(__name__)

class NotificationManager:
    """Gửi thông báo hoàn tất quá trình huấn luyện qua Discord hoặc Telegram Webhook."""

    @classmethod
    def send_discord_notification(cls, webhook_url: str, message: str, likeness_score: Optional[float] = None) -> bool:
        if not webhook_url:
            return False
        payload = {
            "content": f"🎨 **OmniLoRA Studio Alert**:\n{message}\n🏆 **Độ giống đạt được:** `{likeness_score or 0.0}%`"
        }
        try:
            res = requests.post(webhook_url, json=payload, timeout=10)
            return res.status_code == 204
        except Exception as e:
            logger.warning(f"Không thể gửi webhook Discord: {e}")
            return False

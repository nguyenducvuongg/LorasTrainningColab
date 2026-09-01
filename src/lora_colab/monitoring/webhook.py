import os
import json
import requests
from typing import Optional, Dict, Any
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class NotificationManager:
    """Sends training progress updates and sample preview images to Discord or Telegram."""

    @staticmethod
    def send_discord_notification(
        webhook_url: str,
        message: str,
        image_path: Optional[str] = None,
        embed_title: str = "LoRA Training Progress",
        fields: Optional[Dict[str, str]] = None
    ) -> bool:
        """Sends a rich Discord embed with optional image attachment."""
        try:
            embed = {
                "title": embed_title,
                "description": message,
                "color": 0x5865F2,  # Discord Blurple
            }
            if fields:
                embed["fields"] = [{"name": k, "value": str(v), "inline": True} for k, v in fields.items()]

            payload = {"embeds": [embed]}

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files = {
                        "payload_json": (None, json.dumps(payload), "application/json"),
                        "file": (os.path.basename(image_path), f, "image/png")
                    }
                    response = requests.post(webhook_url, files=files, timeout=15)
            else:
                response = requests.post(webhook_url, json=payload, timeout=15)

            if response.status_code in (200, 204):
                logger.info("Discord notification sent successfully.")
                return True
            else:
                logger.warning(f"Discord webhook failed with status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    @staticmethod
    def send_telegram_notification(
        bot_token: str,
        chat_id: str,
        message: str,
        image_path: Optional[str] = None
    ) -> bool:
        """Sends Telegram text or photo notification."""
        try:
            if image_path and os.path.exists(image_path):
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(image_path, "rb") as photo:
                    data = {"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"}
                    files = {"photo": photo}
                    resp = requests.post(url, data=data, files=files, timeout=15)
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
                resp = requests.post(url, data=data, timeout=15)

            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

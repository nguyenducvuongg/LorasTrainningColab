import os
import re
import sys
import time
import base64
from typing import Optional, Dict, Any, List
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class LiveTrainingDashboard:
    """
    Bảng điều khiển giám sát tiến trình huấn luyện LoRA thời gian thực (Real-time Viewheight Dashboard).
    Hiển thị gọn gàng trong 1 khung nhìn màn hình (100vh):
    - Thanh tiến độ kép Step / Epoch + Tốc độ (it/s) + Thời gian còn lại (ETA)
    - Chỉ số thời gian thực: Current Loss, Dynamic LR (Prodigy), VRAM GPU, System RAM
    - Khung xem trước ảnh render mẫu (Inline Live Sample Preview)
    - Cửa sổ Mini-Log Console tự cuộn cố định (140px) không làm tràn màn hình.
    """

    def __init__(
        self,
        model_name: str = "LoRA Model",
        engine_name: str = "AI-Toolkit",
        total_steps: int = 500,
        total_epochs: int = 10,
        output_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.engine_name = engine_name
        self.total_steps = max(1, total_steps)
        self.total_epochs = max(1, total_epochs)
        self.output_dir = output_dir

        self.current_step = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.current_lr = 0.0
        self.speed_str = "-- it/s"
        self.eta_str = "--:--"
        self.status = "Đang khởi tạo..."
        
        self.latest_sample_path: Optional[str] = None
        self.latest_sample_b64: Optional[str] = None
        self.recent_logs: List[str] = []
        self.max_log_lines = 15

        self.start_time = time.time()
        self.last_render_time = 0.0
        self.display_handle = None
        self.is_colab = "google.colab" in sys.modules or "IPython" in sys.modules

        self._init_display()

    def _init_display(self):
        """Khởi tạo display handle trong môi trường Colab/IPython."""
        if self.is_colab:
            try:
                from IPython.display import display, HTML
                self.display_handle = display(HTML(self._render_html()), display_id=True)
            except Exception:
                self.display_handle = None

    def get_hardware_metrics(self) -> Dict[str, str]:
        """Đo lường mức sử dụng VRAM GPU và RAM hệ thống tức thời."""
        vram_str = "N/A"
        ram_str = "N/A"

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_str = f"{allocated:.1f} / {total:.1f} GB"
        except Exception:
            pass

        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_str = f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB"
        except Exception:
            pass

        return {"vram": vram_str, "ram": ram_str}

    def _find_latest_sample_image(self):
        """Quét tìm file ảnh mẫu render mới nhất trong thư mục đầu ra."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return

        image_files = []
        for root, _, files in os.walk(self.output_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    full_p = os.path.join(root, f)
                    try:
                        image_files.append((os.path.getmtime(full_p), full_p))
                    except Exception:
                        pass

        if image_files:
            image_files.sort(key=lambda x: x[0], reverse=True)
            latest = image_files[0][1]
            if latest != self.latest_sample_path:
                self.latest_sample_path = latest
                try:
                    with open(latest, "rb") as img_f:
                        encoded = base64.b64encode(img_f.read()).decode("utf-8")
                        ext = os.path.splitext(latest)[1].replace(".", "").lower()
                        self.latest_sample_b64 = f"data:image/{ext};base64,{encoded}"
                except Exception:
                    pass

    def parse_log_line(self, line: str):
        """Phân tích log stream từ engine để tự động trích xuất Step, Loss, LR, ETA."""
        clean_line = line.strip()
        if not clean_line:
            return

        # Lưu log vào buffer
        timestamp = time.strftime("%H:%M:%S")
        self.recent_logs.append(f"[{timestamp}] {clean_line}")
        if len(self.recent_logs) > self.max_log_lines:
            self.recent_logs.pop(0)

        # 1. Trích xuất Step / Epoch
        step_match = re.search(r"step[s]?\s*[:=]?\s*(\d+)[ /]+(\d+)", clean_line, re.IGNORECASE)
        if step_match:
            try:
                self.current_step = int(step_match.group(1))
                self.total_steps = max(self.total_steps, int(step_match.group(2)))
            except Exception:
                pass

        epoch_match = re.search(r"epoch\s*[:=]?\s*(\d+)[ /]+(\d+)", clean_line, re.IGNORECASE)
        if epoch_match:
            try:
                self.current_epoch = int(epoch_match.group(1))
                self.total_epochs = max(self.total_epochs, int(epoch_match.group(2)))
            except Exception:
                pass

        # 2. Trích xuất Loss
        loss_match = re.search(r"loss\s*[:=]\s*([0-9.]+)", clean_line, re.IGNORECASE)
        if loss_match:
            try:
                self.current_loss = float(loss_match.group(1))
            except Exception:
                pass

        # 3. Trích xuất Learning Rate (LR)
        lr_match = re.search(r"lr\s*[:=]\s*([0-9.eE+-]+)", clean_line, re.IGNORECASE)
        if lr_match:
            try:
                self.current_lr = float(lr_match.group(1))
            except Exception:
                pass

        # 4. Trích xuất tốc độ & ETA
        speed_match = re.search(r"([0-9.]+\s*it/s|[0-9.]+\s*s/it)", clean_line)
        if speed_match:
            self.speed_str = speed_match.group(1)

        eta_match = re.search(r"<([0-9:]+)", clean_line)
        if eta_match:
            self.eta_str = eta_match.group(1)

        # Quét tìm ảnh sample mới
        self._find_latest_sample_image()

        # Cập nhật giao diện (tối đa 2 lần/giây để mượt mà không lag trình duyệt)
        now = time.time()
        if now - self.last_render_time > 0.5:
            self.render()
            self.last_render_time = now

    def _render_html(self) -> str:
        """Tạo HTML5/CSS3 hiện đại, tinh gọn với chiều cao cố định (100vh compact)."""
        pct = min(100, int((self.current_step / max(1, self.total_steps)) * 100))
        metrics = self.get_hardware_metrics()

        lr_display = f"{self.current_lr:.2e}" if self.current_lr > 0 else "Adaptive (Prodigy)"
        loss_display = f"{self.current_loss:.4f}" if self.current_loss > 0 else "Đang tính..."
        
        sample_html = """
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#888; border:2px dashed #444; border-radius:8px; padding:15px; text-align:center;">
            <span style="font-size:24px;">🖼️</span>
            <span style="font-size:12px; margin-top:5px;">Chưa có ảnh render mẫu.<br>Ảnh sẽ hiện ngay khi đến bước sample.</span>
        </div>
        """
        if self.latest_sample_b64:
            filename = os.path.basename(self.latest_sample_path or "sample.png")
            sample_html = f"""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%;">
                <img src="{self.latest_sample_b64}" style="max-height:170px; max-width:100%; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.5); border:1px solid #555; object-fit:contain;" />
                <span style="font-size:11px; color:#aaa; margin-top:4px;">{filename}</span>
            </div>
            """

        logs_escaped = "<br>".join(
            f"<span style='color:{'#ff5555' if 'error' in l.lower() else '#5af78e' if 'step' in l.lower() else '#ddd'};'>{l}</span>"
            for l in self.recent_logs
        ) or "<span style='color:#777;'>Đang chờ nhận log từ engine huấn luyện...</span>"

        html_code = f"""
        <div style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background:#18191c; color:#f0f2f5; border:1px solid #333; border-radius:12px; padding:16px; max-width:950px; margin:10px auto; box-shadow:0 8px 24px rgba(0,0,0,0.4);">
            <!-- HEADER -->
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #2a2b2f; padding-bottom:10px; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:20px;">🎨</span>
                    <div>
                        <span style="font-weight:700; font-size:15px; color:#4ea8de;">Colab LoRA Studio • Live Dashboard</span>
                        <span style="font-size:12px; color:#888; margin-left:8px;">Model: <b style="color:#e0a96d;">{self.model_name}</b> | Engine: <b style="color:#a8dadc;">{self.engine_name}</b></span>
                    </div>
                </div>
                <div style="background:#2a2b2f; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:600; color:#5af78e;">
                    ● LIVE TRAINING
                </div>
            </div>

            <!-- MAIN GRID: METRICS & SAMPLE PREVIEW -->
            <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:14px; margin-bottom:12px;">
                <!-- LEFT: PROGRESS & METRICS -->
                <div style="background:#202225; padding:12px; border-radius:8px; border:1px solid #2e3035;">
                    <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:600; margin-bottom:6px;">
                        <span>Tiến Độ Bước (Step): <b style="color:#4ea8de;">{self.current_step} / {self.total_steps}</b></span>
                        <span style="color:#e0a96d;">{pct}%</span>
                    </div>
                    <!-- PROGRESS BAR -->
                    <div style="background:#333; border-radius:6px; height:12px; overflow:hidden; margin-bottom:10px;">
                        <div style="background:linear-gradient(90deg, #4ea8de, #64dfdf); height:100%; width:{pct}%; transition:width 0.3s ease;"></div>
                    </div>

                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px;">
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">📉 Loss:</span> <b style="color:#f77f00; float:right;">{loss_display}</b>
                        </div>
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">📈 LR:</span> <b style="color:#64dfdf; float:right;">{lr_display}</b>
                        </div>
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">⚡ Tốc độ:</span> <b style="color:#fff; float:right;">{self.speed_str}</b>
                        </div>
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">⏳ ETA:</span> <b style="color:#fff; float:right;">{self.eta_str}</b>
                        </div>
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">💾 VRAM GPU:</span> <b style="color:#a8dadc; float:right;">{metrics['vram']}</b>
                        </div>
                        <div style="background:#18191c; padding:6px 10px; border-radius:6px;">
                            <span style="color:#888;">🧠 Sys RAM:</span> <b style="color:#a8dadc; float:right;">{metrics['ram']}</b>
                        </div>
                    </div>
                </div>

                <!-- RIGHT: SAMPLE IMAGE PREVIEW -->
                <div style="background:#202225; padding:10px; border-radius:8px; border:1px solid #2e3035; height:185px; display:flex; flex-direction:column; justify-content:center;">
                    {sample_html}
                </div>
            </div>

            <!-- BOTTOM: FIXED-HEIGHT AUTO-SCROLL CONSOLE LOG -->
            <div style="background:#121315; border:1px solid #2a2b2f; border-radius:8px; padding:10px; font-family:'Fira Code', 'Courier New', monospace; font-size:11px; line-height:1.4; height:130px; overflow-y:auto; scroll-behavior:smooth;">
                {logs_escaped}
            </div>
        </div>
        """
        return html_code

    def render(self):
        """Cập nhật giao diện trực tiếp lên màn hình Colab."""
        if self.is_colab and self.display_handle:
            try:
                from IPython.display import HTML
                self.display_handle.update(HTML(self._render_html()))
            except Exception:
                pass
        else:
            # Fallback in terminal
            pass

    def close(self, success: bool = True):
        """Hoàn tất quá trình huấn luyện và cập nhật trạng thái cuối cùng."""
        self.status = "Hoàn tất thành công! 🎉" if success else "Đã dừng hoặc gặp lỗi ❌"
        self.render()

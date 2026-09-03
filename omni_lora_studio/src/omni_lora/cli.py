import sys
from pathlib import Path
import click
from .core.config import OmniConfig
from .core.logger import console, setup_logger
from .core.environment import EnvironmentManager
from .dataset.preprocessor import DatasetPreprocessor
from .dataset.face_extractor import FaceAwareCropGenerator
from .dataset.captioning.engine import CaptioningEngine
from .engines.factory import EngineFactory
from .validation.likeness_meter import LikenessMeter

logger = setup_logger(__name__)

@click.group()
def main():
    """🎨 OmniLoRA Studio: Professional All-in-One Multi-Model LoRA Training CLI."""
    pass

@main.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True), help="Đường dẫn file cấu hình YAML")
def train(config_path: str):
    """Bắt đầu chu trình huấn luyện LoRA hoàn chỉnh."""
    console.rule("[bold cyan]OmniLoRA Studio - Training Engine[/bold cyan]")
    config = OmniConfig.from_yaml(config_path)

    # Nếu đang chạy trên Colab, chuẩn bị SSD staging
    if EnvironmentManager.is_colab():
        staged_dataset = EnvironmentManager.prepare_local_staging(config.dataset.dataset_path)
        config.dataset.dataset_path = staged_dataset

    # Khởi tạo trainer và bắt đầu huấn luyện
    trainer = EngineFactory.create_trainer(config)
    success = trainer.run_training()

    if success:
        console.print("[bold green]✅ Huấn luyện thành công hoàn tất![/bold green]")
    else:
        sys.exit(1)

@main.command()
@click.option("-d", "--data-dir", required=True, type=click.Path(exists=True), help="Thư mục ảnh gốc")
@click.option("-o", "--output-dir", required=True, type=click.Path(), help="Thư mục xuất ảnh sạch")
@click.option("--face-crop/--no-face-crop", default=True, help="Bật trích xuất khuôn mặt đa tỷ lệ (100% likeness)")
@click.option("--trigger", default="sks", help="Từ khóa kích hoạt LoRA")
@click.option("--class-name", default="person", help="Lớp chủ thể")
def prep(data_dir: str, output_dir: str, face_crop: bool, trigger: str, class_name: str):
    """Chuẩn hóa ảnh, sửa lỗi EXIF, trích xuất khuôn mặt đa tỷ lệ."""
    console.rule("[bold cyan]OmniLoRA Studio - Tiền Xử Lý Dữ Liệu[/bold cyan]")
    clean_images = DatasetPreprocessor.prepare_clean_dataset(data_dir, output_dir)

    if face_crop:
        console.print("[cyan]🔍 Đang trích xuất tập dữ liệu đa tỷ lệ (Face Close-up, Upper Body, Full Body)...[/cyan]")
        extractor = FaceAwareCropGenerator()
        total_crops = 0
        for img_p in clean_images:
            crops = extractor.process_and_generate_crops(
                image_path=img_p,
                output_dir=output_dir,
                trigger_word=trigger,
                class_word=class_name
            )
            total_crops += len(crops)
        console.print(f"[bold green]✨ Đã tạo thành công {total_crops} ảnh đa tỷ lệ tối ưu cho 100% likeness![/bold green]")

@main.command()
@click.option("-d", "--data-dir", required=True, type=click.Path(exists=True), help="Thư mục ảnh cần gán nhãn")
@click.option("--backend", default="florence2", type=click.Choice(["florence2", "joycaption", "wd14", "gemini"]), help="Backend gán nhãn")
@click.option("--trigger", default="sks", help="Từ khóa kích hoạt LoRA")
@click.option("--class-name", default="person", help="Lớp chủ thể")
@click.option("--isolate/--no-isolate", default=True, help="Bật lọc cô lập chủ thể chống biến dạng mặt")
def caption(data_dir: str, backend: str, trigger: str, class_name: str, isolate: bool):
    """Tự động gán nhãn AI Vision kết hợp bộ lọc cô lập chủ thể bảo vệ nhận diện 100%."""
    console.rule("[bold cyan]OmniLoRA Studio - Auto-Captioning Pipeline[/bold cyan]")
    engine = CaptioningEngine(
        backend=backend,
        trigger_word=trigger,
        class_word=class_name,
        enable_isolation=isolate
    )

    img_files = [p for p in Path(data_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".webp"]]
    console.print(f"[cyan]Đang gán nhãn {len(img_files)} ảnh bằng backend {backend}...[/cyan]")

    for p in img_files:
        cap = engine.process_file(str(p), overwrite=True)
        console.print(f"[dim]{p.name}[/dim] -> [green]{cap[:90]}...[/green]")

    console.print("[bold green]✅ Gán nhãn và cô lập chủ thể hoàn tất 100%![/bold green]")

@main.command()
@click.option("-s", "--sample", required=True, type=click.Path(exists=True), help="Ảnh kiểm nghiệm sinh ra từ LoRA")
@click.option("-g", "--ground-truth-dir", required=True, type=click.Path(exists=True), help="Thư mục ảnh gốc đối chứng")
def eval(sample: str, ground_truth_dir: str):
    """Đo lường độ giống ảnh đầu vào (Likeness Score %) bằng ArcFace Cosine Metric."""
    console.rule("[bold cyan]OmniLoRA Studio - Đánh Giá Độ Giống (100% Likeness Meter)[/bold cyan]")
    meter = LikenessMeter()
    gt_images = [str(p) for p in Path(ground_truth_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".webp"]]
    score = meter.evaluate_sample_against_ground_truth(sample, gt_images)

    console.print(f"\n[bold yellow]📸 Ảnh kiểm thử:[/bold yellow] {sample}")
    console.print(f"[bold green]🏆 ĐIỂM SỐ TƯƠNG ĐỒNG (LIKENESS SCORE):[/bold green] [bold magenta]{score}%[/bold magenta]\n")

if __name__ == "__main__":
    main()

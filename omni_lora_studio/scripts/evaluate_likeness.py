#!/usr/bin/env python3
import sys
from pathlib import Path
import click
from omni_lora.validation.likeness_meter import LikenessMeter
from omni_lora.core.logger import console

@click.command()
@click.option("-s", "--sample", required=True, type=click.Path(exists=True), help="Đường dẫn ảnh sinh từ LoRA")
@click.option("-g", "--ground-truth-dir", required=True, type=click.Path(exists=True), help="Thư mục ảnh gốc đối chứng")
def main(sample: str, ground_truth_dir: str):
    """Đo lường độ giống ảnh đầu vào (Likeness Score %) bằng ArcFace Cosine Metric."""
    console.rule("[bold cyan]OmniLoRA Studio - Standalone Likeness Benchmark[/bold cyan]")
    meter = LikenessMeter()
    gt_images = [str(p) for p in Path(ground_truth_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".webp"]]

    if not gt_images:
        console.print("[bold red]Không tìm thấy ảnh hợp lệ trong thư mục gốc![/bold red]")
        sys.exit(1)

    score = meter.evaluate_sample_against_ground_truth(sample, gt_images)
    console.print(f"📸 Ảnh kiểm nghiệm: [yellow]{sample}[/yellow]")
    console.print(f"📁 Số lượng ảnh gốc đối chứng: [cyan]{len(gt_images)}[/cyan]")
    console.print(f"🏆 [bold green]ĐỘ GIỐNG ĐẦU VÀO (LIKENESS SCORE):[/bold green] [bold magenta]{score}%[/bold magenta]")

if __name__ == "__main__":
    main()

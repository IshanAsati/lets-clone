"""
============================================================
Voice Clone Pipeline - Utilities
============================================================
System checks, GPU info, audio validation.
============================================================
"""

import os
import sys
import shutil
import torch
from rich.console import Console

console = Console()


def check_gpu() -> dict:
    info = {
        "available": torch.cuda.is_available(),
        "device": "cpu",
        "name": "N/A",
        "vram_gb": 0,
        "vram_free_gb": 0,
    }
    if info["available"]:
        info["device"] = "cuda"
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = props.total_mem / (1024 ** 3)
        info["vram_free_gb"] = (props.total_mem - torch.cuda.memory_allocated()) / (1024 ** 3)
    return info


def print_system_info():
    console.print("\n[bold cyan]System Information[/bold cyan]")
    console.print(f"  Python: {sys.version.split()[0]}")
    console.print(f"  PyTorch: {torch.__version__}")
    console.print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"  CUDA version: {torch.version.cuda}")
        gpu = check_gpu()
        console.print(f"  GPU: {gpu['name']}")
        console.print(f"  VRAM: {gpu['vram_gb']:.1f} GB total, {gpu['vram_free_gb']:.1f} GB free")
    console.print(f"  FFmpeg: {bool(shutil.which('ffmpeg'))}")

    # Check toolkit
    toolkit = os.path.isdir("chatterbox-finetuning")
    console.print(f"  Toolkit: {'✓' if toolkit else '✗ (run setup.bat)'}")

    # Check models
    models_dir = os.path.join("chatterbox-finetuning", "pretrained_models")
    models = os.path.isdir(models_dir) and len(os.listdir(models_dir)) > 0 if os.path.isdir(models_dir) else False
    console.print(f"  Pretrained models: {'✓' if models else '✗ (run setup.bat)'}")

    # Check dataset
    dataset = os.path.exists(os.path.join("MyTTSDataset", "metadata.csv"))
    console.print(f"  Dataset: {'✓' if dataset else '✗ (run prepare_dataset.py)'}")

    # Check checkpoints
    checkpoint = any(
        f.endswith(".safetensors")
        for d in ["chatterbox_output", os.path.join("chatterbox-finetuning", "chatterbox_output")]
        if os.path.isdir(d)
        for f in os.listdir(d)
    ) if True else False
    console.print(f"  Trained model: {'✓' if checkpoint else '✗ (run train.py)'}")


if __name__ == "__main__":
    print_system_info()

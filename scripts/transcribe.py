"""
============================================================
Voice Clone Pipeline - Whisper Transcription
============================================================
Transcribes preprocessed clips using Whisper and updates
metadata with transcriptions.

Usage:
    python scripts/transcribe.py
    python scripts/transcribe.py --config config.yaml
============================================================
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("transcribe")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def load_metadata(metadata_path: str) -> list:
    if not os.path.exists(metadata_path):
        console.print(f"[red bold]ERROR:[/] {metadata_path} not found.")
        console.print("  Run: python scripts/preprocess.py")
        sys.exit(1)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_whisper(model_size: str, device: str, fp16: bool):
    """Load faster-whisper (preferred) or openai-whisper."""
    try:
        from faster_whisper import WhisperModel
        compute = "float16" if fp16 and device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute)
        console.print(f"  Engine: [green]faster-whisper[/green] ({model_size}, {compute})")
        return model, "faster"
    except ImportError:
        pass
    try:
        import whisper
        model = whisper.load_model(model_size, device=device)
        console.print(f"  Engine: [green]openai-whisper[/green] ({model_size})")
        return model, "openai"
    except ImportError:
        console.print("[red bold]ERROR:[/] No whisper installed.")
        console.print("  pip install faster-whisper")
        sys.exit(1)


def transcribe_file(model, engine: str, path: str, lang: str, beam: int) -> str:
    if engine == "faster":
        segs, _ = model.transcribe(path, language=lang or None, beam_size=beam, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)
    else:
        result = model.transcribe(path, language=lang or None, beam_size=beam, fp16=torch.cuda.is_available())
        return result["text"].strip()


def transcribe_dataset(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    w_cfg = config["transcription"]
    logger = setup_logging(paths["log_dir"])

    metadata_path = os.path.join("data", "metadata.csv")
    device = w_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠ CUDA unavailable, using CPU[/yellow]")
        device = "cpu"

    console.print(f"\n[bold cyan]═══ Voice Clone Pipeline — Transcription ═══[/bold cyan]")
    console.print(f"  Device: [green]{device}[/green]")

    model, engine = load_whisper(w_cfg["model_size"], device, w_cfg["fp16"])
    metadata = load_metadata(metadata_path)
    console.print(f"  Clips: [green]{len(metadata)}[/green]\n")

    transcribed = 0
    failed = 0
    results = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(metadata))

        for row in metadata:
            clip = row["file"]
            audio_path = row["path"]
            progress.update(task, description=f"Transcribing: {clip}")

            if not os.path.exists(audio_path):
                logger.warning(f"Missing: {audio_path}")
                row["transcript"] = ""
                failed += 1
                progress.advance(task)
                continue

            try:
                text = transcribe_file(model, engine, audio_path, w_cfg.get("language"), w_cfg.get("beam_size", 3))
                # Clean text for LJSpeech format
                text = text.strip().replace("|", " ").replace("\n", " ")
                row["transcript"] = text
                transcribed += 1
                logger.info(f"{clip} → '{text[:60]}'")
            except Exception as e:
                logger.error(f"Failed {clip}: {e}")
                console.print(f"  [red]✗[/red] {clip}: {e}")
                row["transcript"] = ""
                failed += 1

            results.append(row)
            progress.advance(task)

    # Save updated metadata
    fieldnames = ["file", "path", "duration", "sample_rate", "source", "transcript"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    console.print(f"\n[bold green]✓ Transcription Complete[/bold green]")
    console.print(f"  Transcribed: [green]{transcribed}[/green] | Failed: [yellow]{failed}[/yellow]")

    # Preview
    if results:
        table = Table(title="Preview (first 5)", show_lines=True)
        table.add_column("Clip", style="cyan", max_width=25)
        table.add_column("Dur", justify="right", width=6)
        table.add_column("Transcript", max_width=50)
        for row in results[:5]:
            t = row.get("transcript", "")
            table.add_row(row["file"], f"{row['duration']}s", t[:50] + ("..." if len(t) > 50 else ""))
        console.print()
        console.print(table)

    empty = sum(1 for r in results if not r.get("transcript"))
    if empty:
        console.print(f"\n[yellow]⚠ {empty} clip(s) have no transcript. Review and re-record if needed.[/yellow]")

    logger.info(f"Done: {transcribed} OK, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        console.print(f"[red bold]ERROR:[/] {args.config} not found")
        sys.exit(1)
    transcribe_dataset(args.config)

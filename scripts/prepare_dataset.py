"""
============================================================
Voice Clone Pipeline - Prepare Dataset for Training
============================================================
Converts preprocessed + transcribed clips into the LJSpeech
format required by the chatterbox-finetuning toolkit.

Creates:
  MyTTSDataset/
  ├── metadata.csv        # filename|raw_text|normalized_text
  └── wavs/
      ├── recording_001.wav
      └── ...

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --config config.yaml
============================================================
"""

import os
import sys
import csv
import shutil
import argparse
import re
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS training:
    - lowercase
    - remove extra whitespace
    - basic punctuation cleanup
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    # Remove any pipe characters (LJSpeech delimiter)
    text = text.replace("|", " ")
    return text


def prepare_dataset(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    processed_dir = paths["processed_dir"]
    dataset_dir = paths["dataset_dir"]
    wavs_dir = os.path.join(dataset_dir, "wavs")
    metadata_in = os.path.join("data", "metadata.csv")

    console.print(f"\n[bold cyan]═══ Voice Clone Pipeline — Dataset Preparation ═══[/bold cyan]")

    # Check metadata exists
    if not os.path.exists(metadata_in):
        console.print("[red bold]ERROR:[/] data/metadata.csv not found.")
        console.print("  Run: python scripts/preprocess.py && python scripts/transcribe.py")
        sys.exit(1)

    # Read metadata
    with open(metadata_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter rows that have transcripts
    valid_rows = [r for r in rows if r.get("transcript", "").strip()]
    if not valid_rows:
        console.print("[red bold]ERROR:[/] No clips with transcripts found.")
        console.print("  Run: python scripts/transcribe.py")
        sys.exit(1)

    console.print(f"  Total clips: {len(rows)}")
    console.print(f"  With transcripts: [green]{len(valid_rows)}[/green]")
    console.print(f"  Without transcripts: [yellow]{len(rows) - len(valid_rows)}[/yellow] (skipped)")

    # Create dataset directory
    os.makedirs(wavs_dir, exist_ok=True)

    # Copy WAVs and build LJSpeech metadata
    ljspeech_rows = []
    copied = 0
    errors = 0

    for i, row in enumerate(valid_rows):
        src_path = row["path"]
        clip_name = f"recording_{i+1:04d}"
        dst_path = os.path.join(wavs_dir, f"{clip_name}.wav")

        if not os.path.exists(src_path):
            console.print(f"  [red]✗[/red] Missing: {src_path}")
            errors += 1
            continue

        # Copy WAV file
        shutil.copy2(src_path, dst_path)

        # Build LJSpeech entry: filename|raw_text|normalized_text
        raw_text = row["transcript"].strip()
        norm_text = normalize_text(raw_text)

        ljspeech_rows.append({
            "filename": clip_name,
            "raw_text": raw_text,
            "normalized_text": norm_text,
        })
        copied += 1

    # Write LJSpeech metadata.csv
    metadata_out = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_out, "w", newline="", encoding="utf-8") as f:
        for row in ljspeech_rows:
            # LJSpeech format: filename|raw_text|normalized_text (no header, pipe-separated)
            f.write(f"{row['filename']}|{row['raw_text']}|{row['normalized_text']}\n")

    # Also copy best reference audio to speaker_reference/
    ref_dir = paths.get("speaker_reference_dir", "speaker_reference")
    os.makedirs(ref_dir, exist_ok=True)

    ref_dst = os.path.join(ref_dir, "reference.wav")
    if not os.path.exists(ref_dst) and valid_rows:
        # Use the longest clip as reference
        longest = max(valid_rows, key=lambda r: float(r.get("duration", 0)))
        if os.path.exists(longest["path"]):
            shutil.copy2(longest["path"], ref_dst)
            console.print(f"\n  Auto-selected reference audio: {Path(longest['path']).name} ({longest['duration']}s)")
            console.print(f"  Copied to: [cyan]{ref_dst}[/cyan]")

    # Summary
    total_dur = sum(float(r.get("duration", 0)) for r in valid_rows)

    console.print(f"\n[bold green]✓ Dataset Prepared[/bold green]")
    console.print(f"  Clips copied: [green]{copied}[/green]")
    console.print(f"  Errors: [yellow]{errors}[/yellow]")
    console.print(f"  Total audio: [green]{total_dur:.0f}s ({total_dur/60:.1f} min)[/green]")
    console.print(f"  Dataset path: [cyan]{dataset_dir}[/cyan]")
    console.print(f"  Metadata: [cyan]{metadata_out}[/cyan]")

    # Preview
    if ljspeech_rows:
        table = Table(title="LJSpeech Metadata Preview", show_lines=True)
        table.add_column("Filename", style="cyan")
        table.add_column("Raw Text", max_width=40)
        table.add_column("Normalized", max_width=40)
        for row in ljspeech_rows[:5]:
            table.add_row(
                row["filename"],
                row["raw_text"][:40] + ("..." if len(row["raw_text"]) > 40 else ""),
                row["normalized_text"][:40] + ("..." if len(row["normalized_text"]) > 40 else ""),
            )
        console.print()
        console.print(table)

    # Training recommendations
    console.print(f"\n[bold]Training Recommendations:[/bold]")
    if total_dur < 1800:  # < 30 min
        console.print(f"  Dataset: {total_dur/60:.0f} min — [yellow]Small[/yellow]. Train for 100-150 epochs.")
    elif total_dur < 3600:  # < 1 hour
        console.print(f"  Dataset: {total_dur/60:.0f} min — [green]Good[/green]. Train for 75-100 epochs.")
    else:
        console.print(f"  Dataset: {total_dur/60:.0f} min — [green]Excellent[/green]. Train for 50-75 epochs.")

    console.print(f"\n  Next: [cyan]python scripts/train.py[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LJSpeech dataset for training")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        console.print(f"[red bold]ERROR:[/] {args.config} not found")
        sys.exit(1)
    prepare_dataset(args.config)

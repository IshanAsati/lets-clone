"""
============================================================
Voice Clone Pipeline - Audio Preprocessing
============================================================
Converts raw audio → WAV (16kHz mono), splits into clips
(3-10 sec), normalizes, trims silence, generates metadata.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config config.yaml
============================================================
"""

import os
import sys
import csv
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# ── Logging ──────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


# ── FFmpeg Check ─────────────────────────────────────────────
def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        console.print("[red bold]ERROR:[/] FFmpeg is not installed!")
        console.print("  Windows: winget install Gyan.FFmpeg")
        console.print("  Ubuntu:  sudo apt install ffmpeg")
        sys.exit(1)


# ── Audio Loading ────────────────────────────────────────────
def load_audio_file(filepath: str, target_sr: int, mono: bool = True):
    """Load audio file using librosa."""
    try:
        import librosa
        y, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load '{filepath}': {e}")


def save_wav(waveform: np.ndarray, filepath: str, sr: int):
    """Save numpy waveform to WAV."""
    import soundfile as sf
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    sf.write(filepath, waveform, sr, subtype="PCM_16")


# ── Audio Processing ────────────────────────────────────────
def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        return waveform / peak
    return waveform


def trim_silence(waveform: np.ndarray, sr: int, threshold_db: float = -40):
    """Trim leading and trailing silence."""
    import librosa
    trimmed, _ = librosa.effects.trim(waveform, top_db=abs(threshold_db))
    return trimmed


def split_audio(waveform: np.ndarray, sr: int, min_dur: float, max_dur: float):
    """
    Split waveform into clips of [min_dur, max_dur] seconds.
    Uses silence detection for natural split points.
    """
    import librosa

    total_duration = len(waveform) / sr

    if min_dur <= total_duration <= max_dur:
        return [waveform]

    if total_duration < min_dur:
        return []

    clips = []
    intervals = librosa.effects.split(waveform, top_db=30, frame_length=2048, hop_length=512)

    current_clip = np.array([], dtype=waveform.dtype)
    for start, end in intervals:
        segment = waveform[start:end]
        potential = np.concatenate([current_clip, segment]) if len(current_clip) > 0 else segment
        potential_dur = len(potential) / sr

        if potential_dur > max_dur:
            if len(current_clip) / sr >= min_dur:
                clips.append(current_clip)
            current_clip = segment
        else:
            current_clip = potential

    if len(current_clip) > 0 and len(current_clip) / sr >= min_dur:
        clips.append(current_clip)

    # Fallback: hard split
    if len(clips) == 0:
        chunk_samples = int(max_dur * sr)
        for i in range(0, len(waveform), chunk_samples):
            chunk = waveform[i : i + chunk_samples]
            if len(chunk) / sr >= min_dur:
                clips.append(chunk)

    return clips


# ── Find Audio Files ─────────────────────────────────────────
SUPPORTED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus", ".webm", ".mp4"}


def find_audio_files(directory: str) -> list:
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED:
                files.append(os.path.join(root, fname))
    return sorted(files)


# ── Main ─────────────────────────────────────────────────────
def preprocess(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    pre_cfg = config["preprocessing"]
    logger = setup_logging(paths["log_dir"])

    check_ffmpeg()

    raw_dir = paths["raw_audio_dir"]
    out_dir = paths["processed_dir"]
    target_sr = pre_cfg["target_sample_rate"]
    min_dur = pre_cfg["min_clip_duration"]
    max_dur = pre_cfg["max_clip_duration"]

    os.makedirs(out_dir, exist_ok=True)

    audio_files = find_audio_files(raw_dir)
    if not audio_files:
        console.print(f"[red bold]ERROR:[/] No audio files found in '{raw_dir}'")
        console.print(f"  Supported: {', '.join(SUPPORTED)}")
        console.print(f"  Place your voice recordings in '{raw_dir}/' and re-run.")
        sys.exit(1)

    console.print(f"\n[bold cyan]═══ Voice Clone Pipeline — Preprocessing ═══[/bold cyan]")
    console.print(f"  Found [green]{len(audio_files)}[/green] audio file(s) in [yellow]{raw_dir}[/yellow]")
    console.print(f"  Target: {target_sr} Hz mono | Clips: {min_dur}–{max_dur}s\n")

    metadata_rows = []
    clip_count = 0
    skipped = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(audio_files))

        for audio_path in audio_files:
            fname = Path(audio_path).stem
            progress.update(task, description=f"Processing: {Path(audio_path).name}")

            try:
                waveform, sr = load_audio_file(audio_path, target_sr, pre_cfg["mono"])
                logger.info(f"Loaded: {audio_path} ({len(waveform)/sr:.1f}s)")

                if pre_cfg["trim_silence"]:
                    waveform = trim_silence(waveform, sr, pre_cfg["silence_threshold_db"])

                if pre_cfg["normalize"]:
                    waveform = normalize_audio(waveform)

                clips = split_audio(waveform, sr, min_dur, max_dur)

                if not clips:
                    logger.warning(f"Skipped (too short): {audio_path}")
                    skipped += 1
                    progress.advance(task)
                    continue

                for i, clip in enumerate(clips):
                    clip_name = f"{fname}_clip{i:04d}.wav"
                    clip_path = os.path.join(out_dir, clip_name)
                    save_wav(clip, clip_path, sr)

                    duration = len(clip) / sr
                    metadata_rows.append({
                        "file": clip_name,
                        "path": os.path.abspath(clip_path),
                        "duration": round(duration, 2),
                        "sample_rate": sr,
                        "source": Path(audio_path).name,
                    })
                    clip_count += 1

            except Exception as e:
                logger.error(f"Failed: {audio_path}: {e}")
                console.print(f"  [red]✗[/red] {Path(audio_path).name}: {e}")
                skipped += 1

            progress.advance(task)

    # Save metadata
    metadata_path = os.path.join("data", "metadata.csv")
    os.makedirs("data", exist_ok=True)
    if metadata_rows:
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "path", "duration", "sample_rate", "source"])
            writer.writeheader()
            writer.writerows(metadata_rows)

    total_duration = sum(r["duration"] for r in metadata_rows)

    console.print(f"\n[bold green]✓ Preprocessing Complete[/bold green]")
    console.print(f"  Clips: [green]{clip_count}[/green] | Skipped: [yellow]{skipped}[/yellow]")
    console.print(f"  Total duration: [green]{total_duration:.0f}s ({total_duration/60:.1f} min)[/green]")
    console.print(f"  Output: [cyan]{out_dir}[/cyan]")
    console.print(f"  Metadata: [cyan]{metadata_path}[/cyan]")

    if total_duration < 120:
        console.print(
            "\n[yellow]⚠ WARNING:[/yellow] Less than 2 min of audio. "
            "For fine-tuning, aim for [bold]30+ minutes[/bold] (1 hour recommended)."
        )
    elif total_duration < 1800:
        console.print(
            f"\n[dim]  ℹ You have {total_duration/60:.0f} min of audio. "
            "30 min = basic, 60 min = good, 120 min = best quality.[/dim]"
        )

    logger.info(f"Done: {clip_count} clips, {skipped} skipped, {total_duration:.0f}s total")
    return metadata_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio for voice cloning")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        console.print(f"[red bold]ERROR:[/] Config not found: {args.config}")
        sys.exit(1)

    preprocess(args.config)

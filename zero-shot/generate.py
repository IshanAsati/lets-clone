"""
============================================================
Zero-Shot Voice Cloning — Chatterbox TTS
============================================================
Generates speech cloned from a reference audio clip
WITHOUT any training. Just provide a voice sample.

Accuracy: ~70-80% voice similarity
VRAM: ~4-5 GB (fits easily on RTX 4050)

Usage:
    python zero-shot/generate.py --text "Hello world"
    python zero-shot/generate.py --text "Hello" --reference voice.wav
    python zero-shot/generate.py --text "Excited!" --exaggeration 0.8
    python zero-shot/generate.py --file prompts.txt
============================================================
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
from rich.console import Console

console = Console()

# Default reference audio locations (searched in order)
DEFAULT_REFERENCE_PATHS = [
    os.path.join("speaker_reference", "reference.wav"),
    os.path.join("data", "raw", "reference.wav"),
    os.path.join("zero-shot", "reference.wav"),
]


def find_reference() -> str:
    """Find a reference audio file in default locations."""
    for path in DEFAULT_REFERENCE_PATHS:
        if os.path.exists(path):
            return path
    return None


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_speech(
    text: str,
    reference_audio: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    device: str = "cuda",
):
    """Generate speech using Chatterbox zero-shot voice cloning."""

    console.print(f"  Loading Chatterbox TTS model...")
    console.print(f"  (First run downloads ~1-2GB of model weights)")

    # --- Patch: Use toolkit's source (watermarker-free) ---
    toolkit_root = Path(__file__).parent.parent
    toolkit_src = toolkit_root / "chatterbox-finetuning" / "src"
    if toolkit_src.exists():
        if str(toolkit_src) not in sys.path:
            sys.path.insert(0, str(toolkit_src))
        from chatterbox_.tts import ChatterboxTTS
    else:
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            console.print("[red bold]ERROR:[/] chatterbox-tts not installed.")
            console.print("  Install: pip install chatterbox-tts")
            sys.exit(1)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠ CUDA unavailable, using CPU (much slower)[/yellow]")
        device = "cpu"

    try:
        # Prefer our local weights we already downloaded
        local_weights = toolkit_root / "pretrained_models"
        if local_weights.exists():
            model = ChatterboxTTS.from_local(local_weights, device=device)
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
        console.print(f"  [green]✓ Model loaded on {device}[/green]")
    except Exception as e:
        console.print(f"[red bold]ERROR loading model:[/red bold] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

    console.print(f"  Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    console.print(f"  Reference: {Path(reference_audio).name}")
    console.print(f"  Exaggeration: {exaggeration:.2f} | CFG: {cfg_weight:.2f} | Temp: {temperature:.2f}")
    console.print(f"  Generating...")

    try:
        wav = model.generate(
            text,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        # Save output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            torchaudio.save(output_path, wav.cpu(), model.sr)
            duration = wav.shape[-1] / model.sr
        else:
            import soundfile as sf
            sf.write(output_path, wav, model.sr)
            duration = len(wav) / model.sr

        console.print(f"\n  [green]✓ Saved:[/green] {output_path} ({duration:.1f}s)")
        return output_path

    except torch.cuda.OutOfMemoryError:
        console.print("\n[red bold]✗ Out of GPU memory![/red bold]")
        console.print("  Close other GPU-using apps and try again.")
        console.print("  Or try shorter text.")
        clear_gpu()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot voice cloning with Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zero-shot/generate.py --text "Hello world"
  python zero-shot/generate.py --text "Excited!" --exaggeration 0.8
  python zero-shot/generate.py --text "Hello" --reference my_voice.wav
  python zero-shot/generate.py --file prompts.txt
  python zero-shot/generate.py --text "Hello" --output hello.wav
        """,
    )
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--reference", type=str, help="Reference voice audio (6-15s WAV)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion 0.0-1.0 (default: 0.5)")
    parser.add_argument("--cfg", type=float, default=0.5, help="Style adherence 0.0-1.0 (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temp (default: 0.8)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Validate input
    if not args.text and not args.file:
        console.print("[red bold]ERROR:[/] Provide --text or --file")
        parser.print_help()
        sys.exit(1)

    # Find reference audio
    reference = args.reference or find_reference()
    if not reference or not os.path.exists(reference):
        console.print("[red bold]ERROR:[/] No reference audio found!")
        console.print("  Provide one with: --reference path/to/voice.wav")
        console.print("  Or place a file at: speaker_reference/reference.wav")
        console.print("  Or place it at: zero-shot/reference.wav")
        console.print("\n  The reference should be a clean 6-15 second recording of the target voice.")
        sys.exit(1)

    # Collect texts
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        if not os.path.exists(args.file):
            console.print(f"[red bold]ERROR:[/] File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        console.print("[red bold]ERROR:[/] No text to generate.")
        sys.exit(1)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[bold cyan]═══ Zero-Shot Voice Cloning ═══[/bold cyan]")
    console.print(f"  Model: [green]Chatterbox TTS (zero-shot)[/green]")
    console.print(f"  Reference: [cyan]{Path(reference).name}[/cyan]")
    console.print(f"  Texts: [green]{len(texts)}[/green] prompt(s)")
    console.print()

    generated = []
    for i, text in enumerate(texts):
        console.print(f"[bold]── Generating {i+1}/{len(texts)} ──[/bold]")

        if args.output and len(texts) == 1:
            out_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(output_dir, f"zeroshot_{timestamp}_{i:03d}.wav")

        generate_speech(
            text=text,
            reference_audio=reference,
            output_path=out_path,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg,
            temperature=args.temperature,
            device=args.device,
        )
        generated.append(out_path)

    console.print(f"\n[bold green]✓ Done![/bold green]")
    for p in generated:
        console.print(f"  [cyan]{p}[/cyan]")

    console.print(f"\n[dim]  Tips:[/dim]")
    console.print(f"[dim]  • More emotion:   --exaggeration 0.7[/dim]")
    console.print(f"[dim]  • Closer voice:   --cfg 0.7[/dim]")
    console.print(f"[dim]  • More accurate?   Use fine-tuning: python scripts/train.py[/dim]")


if __name__ == "__main__":
    main()

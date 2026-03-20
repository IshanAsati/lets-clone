"""
============================================================
Voice Clone Pipeline - Inference (Fine-Tuned Model)
============================================================
Generates speech using your fine-tuned Chatterbox Turbo model.
Uses the trained .safetensors checkpoint + reference audio
for accurate voice cloning.

Usage:
    python scripts/inference.py --text "Hello, this is my cloned voice."
    python scripts/inference.py --text "I'm excited!" --exaggeration 0.8
    python scripts/inference.py --file prompts.txt
    python scripts/inference.py --text "Hello" --reference my_voice.wav
    python scripts/inference.py --text "Hello" --checkpoint path/to/model.safetensors
============================================================
"""

import os
import sys
import gc
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torchaudio
from rich.console import Console

console = Console()


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_checkpoint(config: dict) -> str:
    """Find the latest fine-tuned model checkpoint."""
    # Check project output dir
    output_dir = "chatterbox_output"
    toolkit_output = os.path.join("chatterbox-finetuning", "chatterbox_output")

    search_dirs = [output_dir, toolkit_output]
    for d in search_dirs:
        if os.path.isdir(d):
            safetensors = glob.glob(os.path.join(d, "*.safetensors"))
            if safetensors:
                # Get the latest one
                latest = max(safetensors, key=os.path.getmtime)
                return latest

    return None


def generate_with_finetuned(
    text: str,
    checkpoint_path: str,
    reference_audio: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    device: str = "cuda",
):
    """
    Generate speech using the fine-tuned Chatterbox model.
    This loads the base model and applies the fine-tuned weights.
    """
    console.print(f"  Loading fine-tuned model...")
    console.print(f"  Checkpoint: [cyan]{Path(checkpoint_path).name}[/cyan]")

    try:
        from chatterbox.tts import ChatterboxTTS

        # Load base model
        model = ChatterboxTTS.from_pretrained(device=device)

        # Load fine-tuned weights
        from safetensors.torch import load_file
        finetuned_weights = load_file(checkpoint_path)

        # Apply fine-tuned T3 weights to the model
        model_state = model.t3.state_dict()
        updated = 0
        for key, value in finetuned_weights.items():
            # Handle key mapping — fine-tuned weights may have different prefixes
            clean_key = key
            for prefix in ["t3.", "model.", "module."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]

            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                model_state[clean_key] = value
                updated += 1

        if updated > 0:
            model.t3.load_state_dict(model_state, strict=False)
            console.print(f"  [green]✓ Applied {updated} fine-tuned weight tensors[/green]")
        else:
            console.print(f"  [yellow]⚠ No matching weights found. Using base model.[/yellow]")
            console.print(f"    This may happen if the checkpoint format is different.")
            console.print(f"    Try using the toolkit's inference.py directly instead.")

        model.eval()

        console.print(f"  Generating speech...")
        console.print(f"  Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        # Generate
        wav = model.generate(
            text,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        # Save
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

        console.print(f"  [green]✓ Saved:[/green] {output_path} ({duration:.1f}s)")
        return output_path

    except torch.cuda.OutOfMemoryError:
        console.print("[red bold]✗ Out of GPU memory![/red bold]")
        console.print("  Close other GPU apps and try again.")
        console.print("  Or try shorter text.")
        clear_gpu()
        sys.exit(1)

    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        console.print(f"\n  If fine-tuned weight loading fails, use the toolkit directly:")
        console.print(f"  [cyan]cd chatterbox-finetuning && python inference.py[/cyan]")
        raise


def generate_with_toolkit(
    text: str,
    reference_audio: str,
    output_path: str,
    toolkit_dir: str = "chatterbox-finetuning",
):
    """
    Fallback: Use the toolkit's own inference.py which handles
    fine-tuned weight loading natively.
    """
    inference_script = os.path.join(toolkit_dir, "inference.py")
    if not os.path.exists(inference_script):
        console.print(f"[red bold]ERROR:[/] {inference_script} not found")
        sys.exit(1)

    # Update the inference script's text and audio prompt
    with open(inference_script, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    # Update TEXT_TO_SAY
    content = re.sub(
        r'(TEXT_TO_SAY\s*=\s*)["\'].*?["\']',
        f'\\g<1>"{text}"',
        content,
    )
    # Update AUDIO_PROMPT
    ref_abs = os.path.abspath(reference_audio).replace("\\", "/")
    content = re.sub(
        r'(AUDIO_PROMPT\s*=\s*)["\'].*?["\']',
        f'\\g<1>"{ref_abs}"',
        content,
    )

    with open(inference_script, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"  Running toolkit inference...")
    console.print(f"  Text: \"{text[:80]}...\"")

    result = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=os.path.abspath(toolkit_dir),
    )

    if result.returncode == 0:
        # Find output file
        toolkit_output = os.path.join(toolkit_dir, "output_stitched.wav")
        if os.path.exists(toolkit_output):
            import shutil
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            shutil.copy2(toolkit_output, output_path)
            console.print(f"  [green]✓ Saved:[/green] {output_path}")
        else:
            console.print(f"  [yellow]⚠ Output file not found at expected location.[/yellow]")
            console.print(f"    Check: {toolkit_dir}/ for output files")
    else:
        console.print(f"[red bold]Inference failed.[/red bold]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with your fine-tuned voice clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py --text "Hello world"
  python scripts/inference.py --text "Excited!" --exaggeration 0.8
  python scripts/inference.py --file prompts.txt
  python scripts/inference.py --text "Hello" --reference my_voice.wav
  python scripts/inference.py --text "Hello" --use-toolkit
        """,
    )
    parser.add_argument("--text", type=str, help="Text to speak")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--reference", type=str, help="Reference audio WAV file")
    parser.add_argument("--checkpoint", type=str, help="Path to .safetensors checkpoint")
    parser.add_argument("--exaggeration", type=float, help="Emotion (0.0-1.0)")
    parser.add_argument("--cfg", type=float, help="CFG weight (0.0-1.0)")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--use-toolkit", action="store_true",
                        help="Use toolkit's inference.py directly (more reliable)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not args.text and not args.file:
        console.print("[red bold]ERROR:[/] Provide --text or --file")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.config):
        console.print(f"[red bold]ERROR:[/] {args.config} not found")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inf_cfg = config["inference"]
    output_dir = config["paths"]["generated_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Reference audio
    reference = args.reference or inf_cfg["reference_audio"]
    if not os.path.exists(reference):
        console.print(f"[red bold]ERROR:[/] Reference audio not found: {reference}")
        console.print("  Place a clean 3-10s WAV file at: speaker_reference/reference.wav")
        console.print("  Or use: --reference path/to/your_voice.wav")
        sys.exit(1)

    # Find checkpoint
    checkpoint = args.checkpoint or find_checkpoint(config)

    # Collect texts
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        if not os.path.exists(args.file):
            console.print(f"[red bold]ERROR:[/] {args.file} not found")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    exaggeration = args.exaggeration if args.exaggeration is not None else inf_cfg["exaggeration"]
    cfg_weight = args.cfg if args.cfg is not None else inf_cfg["cfg_weight"]

    console.print(f"\n[bold cyan]═══ Voice Clone Pipeline — Inference ═══[/bold cyan]")
    console.print(f"  Model: [green]Chatterbox Turbo (Fine-Tuned)[/green]")
    if checkpoint:
        console.print(f"  Checkpoint: [cyan]{Path(checkpoint).name}[/cyan]")
    else:
        console.print(f"  Checkpoint: [yellow]None found — using base model[/yellow]")
    console.print(f"  Reference: [cyan]{Path(reference).name}[/cyan]")
    console.print(f"  Texts: [green]{len(texts)}[/green] prompt(s)")
    console.print(f"  Exaggeration: {exaggeration:.2f} | CFG: {cfg_weight:.2f}")
    console.print()

    generated = []
    for i, text in enumerate(texts):
        console.print(f"[bold]── Generating {i+1}/{len(texts)} ──[/bold]")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.output if (args.output and len(texts) == 1) else \
            os.path.join(output_dir, f"clone_{timestamp}_{i:03d}.wav")

        if args.use_toolkit:
            generate_with_toolkit(text, reference, out_path)
        elif checkpoint:
            generate_with_finetuned(
                text=text,
                checkpoint_path=checkpoint,
                reference_audio=reference,
                output_path=out_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=inf_cfg["temperature"],
                device=inf_cfg["device"],
            )
        else:
            # No checkpoint — use toolkit's inference which handles it natively
            console.print("  [dim]No checkpoint found, using toolkit inference...[/dim]")
            generate_with_toolkit(text, reference, out_path)

        generated.append(out_path)

    console.print(f"\n[bold green]✓ Generation Complete![/bold green]")
    for p in generated:
        console.print(f"  [cyan]{p}[/cyan]")

    console.print(f"\n[dim]  Tips:[/dim]")
    console.print(f"[dim]  • Adjust emotion:     --exaggeration 0.0-1.0[/dim]")
    console.print(f"[dim]  • Adjust style:        --cfg 0.0-1.0[/dim]")
    console.print(f"[dim]  • Use toolkit directly: --use-toolkit (most reliable)[/dim]")


if __name__ == "__main__":
    main()

"""
============================================================
Voice Clone Pipeline - Training (Chatterbox Turbo Fine-Tuning)
============================================================
Fine-tunes the Chatterbox Turbo model on your prepared dataset.
Uses the gokhaneraslan/chatterbox-finetuning toolkit.

Optimized for RTX 4050 (6GB VRAM):
  - batch_size=2, grad_accum=32, fp16=True
  - Effective batch size = 64

Prerequisites:
  1. Run setup.bat/setup.sh first (clones toolkit + downloads models)
  2. Run preprocess.py → transcribe.py → prepare_dataset.py
  3. Dataset should be in MyTTSDataset/ in LJSpeech format

Usage:
    python scripts/train.py
    python scripts/train.py --config config.yaml
    python scripts/train.py --epochs 100
    python scripts/train.py --batch-size 1  (if OOM with batch_size=2)
============================================================
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def check_toolkit():
    """Verify chatterbox-finetuning toolkit is present."""
    toolkit_dir = "chatterbox-finetuning"
    if not os.path.isdir(toolkit_dir):
        console.print("[red bold]ERROR:[/] chatterbox-finetuning toolkit not found!")
        console.print("  Run setup.bat (Windows) or setup.sh (Linux) first.")
        console.print("  Or manually: git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git")
        sys.exit(1)

    required = [
        os.path.join(toolkit_dir, "train.py"),
        os.path.join(toolkit_dir, "src", "config.py"),
        os.path.join(toolkit_dir, "setup.py"),
    ]
    for f in required:
        if not os.path.exists(f):
            console.print(f"[red bold]ERROR:[/] Missing: {f}")
            console.print("  Re-clone the toolkit.")
            sys.exit(1)

    return toolkit_dir


def check_pretrained_models(toolkit_dir: str):
    """Verify pretrained models have been downloaded."""
    model_dir = os.path.join(toolkit_dir, "pretrained_models")
    if not os.path.isdir(model_dir):
        console.print("[red bold]ERROR:[/] Pretrained models not found!")
        console.print(f"  Run: cd {toolkit_dir} && python setup.py")
        sys.exit(1)

    required_models = ["ve.safetensors", "s3gen.safetensors", "t3_cfg.safetensors", "tokenizer.json"]
    # Check for turbo variant too
    has_turbo = os.path.exists(os.path.join(model_dir, "t3_turbo_v1.safetensors"))

    missing = []
    for m in required_models:
        # t3_cfg.safetensors might be t3_turbo_v1.safetensors in turbo mode
        if m == "t3_cfg.safetensors" and has_turbo:
            continue
        if not os.path.exists(os.path.join(model_dir, m)):
            missing.append(m)

    if missing:
        console.print(f"[yellow]⚠ Missing model files: {', '.join(missing)}[/yellow]")
        console.print(f"  Run: cd {toolkit_dir} && python setup.py")
        return False

    console.print(f"  [green]✓[/green] Pretrained models found")
    return True


def check_dataset():
    """Verify dataset is prepared."""
    dataset_dir = "MyTTSDataset"
    metadata = os.path.join(dataset_dir, "metadata.csv")
    wavs_dir = os.path.join(dataset_dir, "wavs")

    if not os.path.exists(metadata):
        console.print("[red bold]ERROR:[/] MyTTSDataset/metadata.csv not found!")
        console.print("  Run: python scripts/prepare_dataset.py")
        sys.exit(1)

    if not os.path.isdir(wavs_dir):
        console.print("[red bold]ERROR:[/] MyTTSDataset/wavs/ not found!")
        sys.exit(1)

    # Count entries
    with open(metadata, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    num_clips = len(lines)

    wav_files = [f for f in os.listdir(wavs_dir) if f.endswith(".wav")]

    console.print(f"  [green]✓[/green] Dataset: {num_clips} entries, {len(wav_files)} WAV files")

    if num_clips < 10:
        console.print(f"[yellow]⚠ Only {num_clips} clips. Aim for 100+ for best results.[/yellow]")

    return num_clips


def update_toolkit_config(toolkit_dir: str, config: dict, args):
    """
    Update the chatterbox-finetuning src/config.py with our settings.
    This patches the config file with our training parameters.
    """
    config_file = os.path.join(toolkit_dir, "src", "config.py")

    if not os.path.exists(config_file):
        console.print(f"[red bold]ERROR:[/] {config_file} not found")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()

    train_cfg = config["training"]
    model_cfg = config["model"]

    # Determine values (CLI args override config)
    batch_size = args.batch_size if args.batch_size else train_cfg["batch_size"]
    epochs = args.epochs if args.epochs else train_cfg["num_epochs"]
    lr = train_cfg["learning_rate"]
    grad_accum = train_cfg["grad_accumulation_steps"]
    is_turbo = model_cfg["is_turbo"]

    # Build replacement map (targets lowercase keys in src/config.py)
    import re

    replacements = {
        r'(batch_size[:\s]*int\s*=\s*)\d+': f'\\g<1>{batch_size}',
        r'(num_epochs[:\s]*int\s*=\s*)\d+': f'\\g<1>{epochs}',
        r'(grad_accum[:\s]*int\s*=\s*)\d+': f'\\g<1>{grad_accum}',
        r'(learning_rate[:\s]*float\s*=\s*)[0-9e.\-]+': f'\\g<1>{lr}',
        r'(is_turbo[:\s]*bool\s*=\s*)(True|False)': f'\\g<1>{is_turbo}',
    }

    # Also set dataset paths (using forward slashes for Windows compatibility)
    dataset_abs = os.path.abspath("MyTTSDataset").replace("\\", "/")
    replacements[r'(csv_path[:\s]*str\s*=\s*)["\'].*?["\']'] = f'\\g<1>"{dataset_abs}/metadata.csv"'
    replacements[r'(wav_dir[:\s]*str\s*=\s*)["\'].*?["\']'] = f'\\g<1>"{dataset_abs}/wavs"'
    replacements[r'(preprocessed_dir\s*=\s*)["\'].*?["\']'] = f'\\g<1>"{dataset_abs}/preprocess"'

    for pattern, replacement in replacements.items():
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        if new_content != content:
            content = new_content

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"  [green]✓[/green] Config updated: batch={batch_size}, epochs={epochs}, lr={lr}, turbo={is_turbo}")
    console.print(f"    Effective batch size: {batch_size * grad_accum}")


def link_dataset(toolkit_dir: str):
    """
    Ensure the toolkit can find the dataset.
    Creates a symlink or copies the dataset into the toolkit directory.
    """
    src = os.path.abspath("MyTTSDataset")
    dst = os.path.join(toolkit_dir, "MyTTSDataset")

    if os.path.exists(dst):
        # The toolkit comes with its own empty MyTTSDataset — remove it!
        # Check if it's our link or their folder
        if os.path.islink(dst):
            return
        else:
            console.print(f"  [dim]Removing toolkit's default dataset folder...[/dim]")
            shutil.rmtree(dst)

    try:
        # Try symlink first (works on Windows with dev mode or admin)
        os.symlink(src, dst, target_is_directory=True)
        console.print(f"  [green]✓[/green] Dataset linked to toolkit")
    except (OSError, NotImplementedError):
        # Fallback: copy the dataset
        console.print(f"  [dim]Copying dataset to toolkit (symlink failed)...[/dim]")
        shutil.copytree(src, dst)
        console.print(f"  [green]✓[/green] Dataset copied to toolkit")


def run_training(toolkit_dir: str):
    """Run the actual training via the toolkit's train.py."""
    train_script = os.path.join(toolkit_dir, "train.py")

    console.print(f"\n[bold green]═══ Starting Training ═══[/bold green]")
    console.print(f"  Script: {train_script}")
    console.print(f"  Working dir: {os.path.abspath(toolkit_dir)}")
    console.print(f"  Output: chatterbox_output/")
    console.print(f"\n  [dim]Press Ctrl+C to stop (checkpoints are saved each epoch)[/dim]\n")
    console.print("─" * 60)

    try:
        # Set memory optimization env vars
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=os.path.abspath(toolkit_dir),
            env=env,
        )

        if result.returncode == 0:
            console.print(f"\n[bold green]✓ Training Complete![/bold green]")

            # Find output
            output_dir = os.path.join(toolkit_dir, "chatterbox_output")
            if os.path.isdir(output_dir):
                safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
                if safetensors:
                    latest = sorted(safetensors)[-1]
                    console.print(f"  Model saved: [cyan]{os.path.join(output_dir, latest)}[/cyan]")

                    # Copy to our output dir
                    our_output = os.path.abspath("chatterbox_output")
                    os.makedirs(our_output, exist_ok=True)
                    for sf in safetensors:
                        src = os.path.join(output_dir, sf)
                        dst = os.path.join(our_output, sf)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                    console.print(f"  Also copied to: [cyan]{our_output}[/cyan]")
        else:
            console.print(f"\n[red bold]Training failed with exit code {result.returncode}[/red bold]")
            console.print("  Check the error output above for details.")
            console.print("  Common fixes:")
            console.print("    - OOM: try --batch-size 1")
            console.print("    - Missing models: cd chatterbox-finetuning && python setup.py")

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Training stopped by user.[/yellow]")
        console.print(f"  Checkpoints saved in: chatterbox-finetuning/chatterbox_output/")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Chatterbox Turbo on your voice dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size (use 1 if OOM)")
    parser.add_argument("--check-only", action="store_true", help="Only verify setup, don't train")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        console.print(f"[red bold]ERROR:[/] {args.config} not found")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    console.print(f"\n[bold cyan]═══ Voice Clone Pipeline — Training ═══[/bold cyan]")
    console.print(f"  Model: [green]Chatterbox {'Turbo' if config['model']['is_turbo'] else 'Standard'}[/green]")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            console.print(f"  GPU: [green]{name} ({vram:.1f} GB)[/green]")
        else:
            console.print("  GPU: [red]None (CPU only — training will be very slow)[/red]")
    except:
        pass

    # Checks
    toolkit_dir = check_toolkit()
    check_pretrained_models(toolkit_dir)
    num_clips = check_dataset()

    if args.check_only:
        console.print("\n[green]✓ All checks passed. Ready to train.[/green]")
        return

    # Update config
    update_toolkit_config(toolkit_dir, config, args)

    # Link dataset
    link_dataset(toolkit_dir)

    # Training info
    train_cfg = config["training"]
    bs = args.batch_size or train_cfg["batch_size"]
    ep = args.epochs or train_cfg["num_epochs"]
    ga = train_cfg["grad_accumulation_steps"]

    console.print(f"\n  [bold]Training Configuration:[/bold]")
    console.print(f"    Batch size:      {bs}")
    console.print(f"    Grad accum:      {ga}")
    console.print(f"    Effective batch: {bs * ga}")
    console.print(f"    Epochs:          {ep}")
    console.print(f"    Learning rate:   {train_cfg['learning_rate']}")
    console.print(f"    FP16:            {train_cfg['fp16']}")
    console.print(f"    Dataset clips:   {num_clips}")

    if bs >= 4:
        console.print(f"\n  [yellow]⚠ batch_size={bs} may OOM on 6GB VRAM. Use --batch-size 2 or 1.[/yellow]")

    # Start
    run_training(toolkit_dir)

    console.print(f"\n  Next: [cyan]python scripts/inference.py --text \"Hello world\"[/cyan]")


if __name__ == "__main__":
    main()

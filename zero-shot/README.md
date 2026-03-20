# Zero-Shot Voice Cloning (Chatterbox TTS)

Quick voice cloning **without any training**. Just provide a 6-15 second reference audio clip and generate speech in that voice immediately.

## Accuracy

~70-80% voice similarity (compared to ~90-95% with fine-tuning).

**Use this when:**
- You want instant results with no training
- You don't have much voice data
- You want a quick test before committing to fine-tuning

## Usage

```bash
# From the project root (with venv activated)
python zero-shot/generate.py --text "Hello, this is my cloned voice."

# With a specific reference audio
python zero-shot/generate.py --text "Hello" --reference path/to/voice.wav

# Adjust emotion (0.0 = neutral, 1.0 = max)
python zero-shot/generate.py --text "I'm excited!" --exaggeration 0.8

# Adjust voice adherence (0.0 = loose, 1.0 = strict)
python zero-shot/generate.py --text "Hello" --cfg 0.7

# Batch mode (one text per line)
python zero-shot/generate.py --file prompts.txt

# Custom output
python zero-shot/generate.py --text "Hello" --output my_output.wav
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exaggeration` | 0.5 | Emotion intensity (0.0-1.0) |
| `--cfg` | 0.5 | How closely to match reference voice style |
| `--temperature` | 0.8 | Randomness (lower = more consistent) |

## Tips

- **Reference audio quality matters most** — use a clean, clear recording
- Best reference length: **6-15 seconds** of continuous speech
- No background music or noise in the reference
- Try different `--exaggeration` and `--cfg` values to find the sweet spot
- For longer texts, the script auto-splits into sentences for better quality

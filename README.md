# 🎙️ Voice Clone Pipeline (RTX 4050 Optimized)

Welcome! This repository is a "Self-Service Voice Lab." It allows you to clone any voice (including your own!) with extremely high accuracy using professional AI models like **Chatterbox Turbo** and **XTTS v2**.

---

## 🚀 Beginner's Quick Start (The Workflow)

If you have never cloned a voice before, just follow these **4 main steps** in order. 

> [!IMPORTANT]
> Always use `.venv\Scripts\python.exe` to run your commands! This ensures your AI tools can see the GPU.

### 1. The Setup (Run Once)
Double-click `setup.bat` or run:
```ps1
.\setup.bat
```
*This downloads the library, the AI "brains" (2GB+), and prepares your GPU.*

### 2. Add your Voice
Put your audio recordings (WAV, MP3, or M4A) into the `data/raw/` folder.
*Tip: 15 minutes of clear audio is plenty for a great start!*

### 3. Prepare the Data
Run these three commands in order. They chop your audio, transcribe it to text, and build the dataset:
```ps1
.venv\Scripts\python.exe scripts/preprocess.py
.venv\Scripts\python.exe scripts/transcribe.py
.venv\Scripts\python.exe scripts/prepare_dataset.py
```

### 4. Deep Fine-Tuning (The Magic)
Start the training. This is when the AI "listens" to your voice over and over to learn it:
```ps1
.venv\Scripts\python.exe scripts/train.py
```
*Wait for it to finish (takes 2-4 hours). The trained model will appear in `chatterbox_output/`.*

---

## 🧠 Which Model Should I Use?

| Model | Speed | Emotion | Best For... |
|-------|-------|---------|-------------|
| **Chatterbox Turbo** | ⚡ Fast | Good | Real-time apps, quick projects, 6GB GPUs. |
| **XTTS v2** | 🐢 Slower | ✨ Elite | High-fidelity acting, whispering, or complex emotions. |

*To use XTTS, look inside the `xtts/` folder for separate instructions.*

---

## 📁 What are these folders?

- **`data/raw/`**: 📥 **YOUR** audio goes here first.
- **`data/processed/`**: ✂️ The AI chops your audio into small clips here.
- **`MyTTSDataset/`**: 📝 The final dataset with transcripts (ready for training).
- **`chatterbox_output/`**: 🏆 Your final, trained "Voice Model" lives here.
- **`outputs/`**: 🔊 Your generated human-like speech files (.wav).
- **`zero-shot/`**: ⚡ "Instant" cloning folder. No training needed, lower quality (~75%).

---

## 🛠️ Optimizations for RTX 4050 (6GB VRAM)

Your GPU is powerful but has a tight 6GB memory limit. We use these "Deep Learning Hacks" automatically to fit:

1.  **Gradient Checkpointing:** Saves ~40% memory by re-calculating small parts of the math only when needed.
2.  **Effective Batching:** We use a `batch_size: 2` and `grad_accum: 16`. This means the AI thinks it's looking at 32 voices at once while only holding 2 in memory!
3.  **FP16:** We use 16-bit math instead of 32-bit. It's twice as fast and uses half the memory.

---

## ❓ Common Beginner Questions

**"I get 'ModuleNotFoundError' when running a script!"**
- Make sure you are using `.venv\Scripts\python.exe` followed by the script name. If you just type `python`, your computer might try to use a different version.

**"Training says 8 hours remaining!"**
- The first epoch is always slow. It usually speeds up. If it's still too slow, edit `config.yaml` and set `num_epochs: 50`.

**"My GPU VRAM is full (OOM)!"**
- Close Chrome, Discord, and games. They eat your 6GB VRAM! If it's still full, set `batch_size: 1` in `config.yaml`.

---

## ⚖️ Legal & Ethical Use
By using this repo, you agree to only clone voices you have the legal right or permission to clone. Use responsibly!

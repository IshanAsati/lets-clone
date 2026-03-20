import os
import wave
import numpy as np
import pytest
from pathlib import Path

@pytest.fixture
def dummy_wav(tmp_path):
    """Generate a 1-second 16kHz mono silent WAV file."""
    path = tmp_path / "test_dummy.wav"
    sample_rate = 16000
    n_samples = sample_rate
    samples = np.zeros(n_samples, dtype=np.int16)
    
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return path

@pytest.fixture
def root_dir():
    return Path(__file__).parent.parent

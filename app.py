import io
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicGenerator(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def generate_harmonic_wave(frequency, duration, sample_rate=44100):
    """Generate a harmonic wave with some musical characteristics."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more musical waveform with multiple harmonics
    wave = (
        np.sin(2 * np.pi * frequency * t) +  # Fundamental frequency
        0.5 * np.sin(2 * np.pi * (frequency * 2) * t) +  # Second harmonic
        0.25 * np.sin(2 * np.pi * (frequency * 3) * t)  # Third harmonic
    )
    
    # Apply envelope to reduce harshness
    envelope = np.exp(-t * 5)
    wave *= envelope
    
    return wave

def generate_music(prompt="", duration=3, sample_rate=44100):
    # Map prompt to musical characteristics
    prompt_seed = hash(prompt) % 1000
    np.random.seed(prompt_seed)
    
    # Select a base frequency based on prompt
    base_frequencies = [
        220,  # A3 - soft
        261.63,  # C4 - neutral
        329.63,  # E4 - bright
        392,  # G4 - warm
    ]
    
    # Choose frequency based on prompt length or content
    frequency = base_frequencies[len(prompt) % len(base_frequencies)]
    
    # Generate musical wave
    audio = generate_harmonic_wave(frequency, duration, sample_rate)
    
    # Add some gentle variation
    variation = np.random.normal(0, 0.1, audio.shape)
    audio += variation
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    # Save to buffer
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    
    return buffer

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_music_endpoint(prompt: str = "calm melody", duration: int = 3):
    buffer = generate_music(prompt, duration)
    return StreamingResponse(buffer, media_type="audio/wav")

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
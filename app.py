import io
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Neural Network for Music Generation
class SimpleMusicGenerator(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the model and create a simple generation function
model = SimpleMusicGenerator()

def generate_music(prompt="", duration=3, sample_rate=44100):
    # Generate a random seed based on input
    seed = hash(prompt) % 1000
    torch.manual_seed(seed)
    
    # Generate random input
    input_tensor = torch.randn(1, 100)
    
    # Generate audio-like output
    with torch.no_grad():
        output = model(input_tensor)
    
    # Create a simple waveform
    audio = np.sin(np.linspace(0, duration * 2 * np.pi * 440, int(duration * sample_rate)))
    audio += output.numpy().flatten()[:len(audio)]
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    # Save to buffer
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    
    return buffer

# Create FastAPI app
app = FastAPI()

@app.post("/generate")
async def generate_music_endpoint(prompt: str = "calm melody", duration: int = 3):
    buffer = generate_music(prompt, duration)
    return StreamingResponse(buffer, media_type="audio/wav")

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

# requirements.txt content:
# fastapi
# uvicorn
# torch
# numpy
# scipy
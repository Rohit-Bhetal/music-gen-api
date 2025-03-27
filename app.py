import io
import torch
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load model (only once when the server starts)
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

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

def generate_music(prompt: str = "calm piano melody", duration: int = 4):
    try:
        # Prepare inputs
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        
        # Generate audio
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=duration * 50)
        
        # Convert to numpy and prepare for wav
        audio_numpy = audio_values[0].cpu().numpy()
        
        # Normalize
        audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))
        audio_numpy = (audio_numpy * 32767).astype(np.int16)
        
        # Save to buffer
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, 32000, audio_numpy)
        buffer.seek(0)
        
        return buffer
    
    except Exception as e:
        print(f"Error generating music: {e}")
        return None

@app.post("/generate")
async def generate_music_endpoint(prompt: str = "calm piano", duration: int = 4):
    buffer = generate_music(prompt, duration)
    if buffer:
        return StreamingResponse(buffer, media_type="audio/wav")
    else:
        return {"error": "Could not generate music"}

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
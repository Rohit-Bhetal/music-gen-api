from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import torch
from audiocraft.models import MusicGen
import io
import scipy.io.wavfile
import numpy as np
import uvicorn

app = FastAPI(title="MusicGen API", description="Generate music with MusicGen Small")

# Load the smallest MusicGen model (300M parameters)
model = MusicGen.get_pretrained('facebook/musicgen-small')
SAMPLE_RATE = model.sample_rate

# Define request model
class MusicRequest(BaseModel):
    prompt: str = "A calm acoustic melody"
    duration: float = 10.0
    temperature: float = 1.0
    stream: bool = False

def generate_audio_chunks(prompt: str, duration: float, temperature: float):
    """Generate audio and yield it in chunks for streaming."""
    duration = min(max(float(duration), 5), 30)  # Clamp between 5 and 30 seconds
    model.set_generation_params(duration=duration, temperature=temperature)

    # Generate audio
    wav = model.generate([prompt], progress=True)
    audio = wav[0].cpu().numpy()

    # Create in-memory buffer and write WAV data
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=SAMPLE_RATE, data=audio)
    buffer.seek(0)

    # Yield audio in chunks (64KB)
    chunk_size = 64 * 1024
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk
    buffer.close()

@app.post("/generate")
async def generate_music(request: MusicRequest):
    """Generate music based on request parameters."""
    # Validate temperature
    temperature = min(max(request.temperature, 0.5), 2.0)  # Clamp between 0.5 and 2.0

    try:
        if request.stream:
            # Stream the audio
            return StreamingResponse(
                generate_audio_chunks(request.prompt, request.duration, temperature),
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=generated_music.wav"}
            )
        else:
            # Generate and return full file for download
            duration = min(max(request.duration, 5), 30)
            model.set_generation_params(duration=duration, temperature=temperature)
            wav = model.generate([request.prompt], progress=True)
            audio = wav[0].cpu().numpy()

            buffer = io.BytesIO()
            scipy.io.wavfile.write(buffer, rate=SAMPLE_RATE, data=audio)
            buffer.seek(0)

            return FileResponse(
                buffer,
                media_type="audio/wav",
                filename="generated_music.wav",
                headers={"Content-Disposition": "attachment; filename=generated_music.wav"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
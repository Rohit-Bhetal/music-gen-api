from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, BarkModel
import numpy as np
import scipy.io.wavfile
import io

app = FastAPI()

# Load Bark model (small version, CPU-only to save memory)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to("cpu")

def generate_audio(prompt: str):
    inputs = processor(prompt, return_tensors="pt")
    audio = model.generate(**inputs, do_sample=True).cpu().numpy()
    audio = audio.squeeze()  # Remove extra dimensions
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=24000, data=audio.astype(np.float32))
    buffer.seek(0)
    return buffer

@app.post("/generate")
async def generate_music(prompt: str = "A calm acoustic melody"):
    buffer = generate_audio(prompt)
    return StreamingResponse(buffer, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
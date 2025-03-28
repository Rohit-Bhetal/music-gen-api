import io
import torch
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Global model and processor variables
model = None
processor = None

def load_model():
    """Lazily load the Musicgen model"""
    global model, processor
    if model is None:
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.float16  # Use half precision to reduce memory
        )
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        # Move to CPU to reduce GPU memory usage
        model = model.to("cpu")

# Create FastAPI app
app = FastAPI()

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "https://capstone-project-ai-saas.vercel.app/",  # Replace with your frontend domain
        # Add other allowed origins
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Specific methods
    allow_headers=["*"]  # Or specify exact headers if needed
)

def generate_music(prompt: str = "calm piano melody", duration: int = 4):
    """Generate music from a text prompt"""
    try:
        # Lazy load model
        load_model()
       
        # Prepare inputs
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
       
        # Generate audio with memory efficiency
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 50,
                do_sample=True,
                temperature=1.0
            )
       
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_music_endpoint(request: Request, prompt: str = "calm piano", duration: int = 4):
    """Endpoint for music generation"""
    buffer = generate_music(prompt, duration)
    return StreamingResponse(
        buffer, 
        media_type="audio/wav"
    )

@app.options("/generate")
async def options_handler():
    """Handle preflight CORS requests"""
    return JSONResponse(status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
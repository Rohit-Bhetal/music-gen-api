import io
import torch
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load model in a memory-efficient way
model = None
processor = None

def load_model():
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"] 
)

def generate_music(prompt: str = "calm piano melody", duration: int = 4):
    try:
        # Lazy load model to reduce initial memory footprint
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
                do_sample=True,  # Add some randomness
                temperature=1.0  # Control creativity
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
        return None
@app.post("/generate")
async def generate_music_endpoint(request: Request, prompt: str = "calm piano", duration: int = 4):
    # Debug: Print incoming headers
    print("Incoming headers:", request.headers)
    
    buffer = generate_music(prompt, duration)
    if buffer:
        response = StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        # Debug: Print response headers
        print("Response headers:", response.headers)
        return response
    else:
        return JSONResponse(
            content={"error": "Could not generate music"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
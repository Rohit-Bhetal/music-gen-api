from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf
import io
import os
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained model (you would train this separately)
model = MusicGenerator()
model.load_state_dict(torch.load('music_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/generate', methods=['GET'])
def generate():
    try:
        # Generate seed (you might want a more sophisticated seed generation)
        seed = np.random.rand(1, 10, 128).astype(np.float32)
        
        # Generate music
        music = generate_music(model, seed)
        
        # Save to in-memory file
        buffer = io.BytesIO()
        sf.write(buffer, music, 44100, format='WAV')
        buffer.seek(0)
        
        return send_file(
            buffer, 
            mimetype='audio/wav', 
            as_attachment=True, 
            download_name='generated_music.wav'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import soundfile as sf
import os

class MusicGenerator(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=128, num_layers=2):
        super(MusicGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for generating musical sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=0.3)
        
        # Fully connected layer to map LSTM output to music notes
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # If no hidden state provided, initialize it
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # Pass through LSTM
        out, hidden = self.lstm(x, hidden)
        
        # Generate output
        out = self.fc(out)
        return out, hidden

# generate_music.py
def generate_music(model, seed, duration=5, sample_rate=44100):
    """
    Generate a music track using the trained model
    
    Args:
    - model: Trained MusicGenerator model
    - seed: Initial seed for generation
    - duration: Length of music in seconds
    - sample_rate: Audio sample rate
    
    Returns:
    - NumPy array of generated music
    """
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    # Prepare seed
    seed = torch.FloatTensor(seed).unsqueeze(0).to(device)
    
    # Number of samples to generate
    num_samples = int(duration * sample_rate)
    
    # Generate music sequence
    generated_sequence = []
    hidden = None
    
    for _ in range(num_samples):
        with torch.no_grad():
            output, hidden = model(seed, hidden)
        
        # Sample from the output distribution
        probabilities = F.softmax(output[0, -1], dim=0)
        note = torch.multinomial(probabilities, 1).item()
        
        # Convert note to audio-like representation
        generated_note = np.sin(2 * np.pi * note * np.linspace(0, 1, 100))
        generated_sequence.extend(generated_note)
        
        # Prepare next input
        seed = torch.FloatTensor([note]).unsqueeze(0).unsqueeze(0).to(device)
    
    return np.array(generated_sequence)
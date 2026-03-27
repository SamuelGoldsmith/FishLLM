import requests
import numpy as np
import soundfile as sf
import io

response = requests.get(
    "http://localhost:8000/tts?prompt=" + input("prumpt: "),
    stream=True,
    timeout=30
)

audio = b""
try:
    for chunk in response.iter_content(chunk_size=512):
        if chunk:
            audio += chunk
except Exception as e:
    print(f"Stream error: {e}")

print(f"Received {len(audio)} bytes")

# Convert raw 16-bit PCM bytes to numpy array and save as MP3
samples = np.frombuffer(audio, dtype=np.int16)
sf.write("output.mp3", samples, samplerate=24000)

print("Saved output.mp3 - open it in any audio player")
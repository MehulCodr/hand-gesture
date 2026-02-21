# %%
import os
import google.generativeai as genai
import pathlib
import tempfile
import pyaudio
import webrtcvad
import numpy as np
import sounddevice as sd

genai.configure(api_key="AIzaSyBdpCS6P8xdphp7768Jy6AclyjJpuGcvGs")

# %%
model = genai.GenerativeModel('gemini-1.5-flash')
prompt = "Generate a transcript of the speech."

def stt():
    response = model.generate_content([
        prompt,
        {
            "mime_type": "audio/ogg",
            "data": pathlib.Path('yappin.ogg').read_bytes()
        }
    ])
    return response.text

# %%
# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # 1 is low, 2 is medium, 3 is high sensitivity

# Function to process audio and return transcript
def process_audio_chunk(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
        temp_file.write(audio_data)
        response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/ogg",
                "data": pathlib.Path(temp_file.name).read_bytes()
            }
        ])
    return response.text

# Function to execute actions based on commands
def execute_command(transcript):
    if "turn on the lights" in transcript:
        print("Turning on the lights!")
        # Add your logic to turn on the lights
    elif "play music" in transcript:
        print("Playing music!")
        # Add your logic to play music

# Function to check if audio contains speech
def is_speech(frame, sample_rate):
    frame_length = 320  # 20 ms of audio for 16kHz
    if len(frame) != frame_length:
        raise ValueError(f"Expected frame length {frame_length}, but got {len(frame)}")
    return vad.is_speech(frame, sample_rate)

# Continuous listening
def listen_and_process():
    rate = 16000  # 16 kHz sample rate
    chunk_size = 320  # 20 ms of audio, 320 samples at 16 kHz
    silence_duration = 0
    silence_threshold = 5  # seconds of silence to stop recording
    audio_frames = []  # renamed to avoid conflict

    def callback(indata, frames, time, status):
        nonlocal silence_duration, audio_frames
        if status:
            print(status)

        # Convert audio data to an array (16-bit PCM)
        audio_array = np.frombuffer(indata, dtype=np.int16)

        # Ensure the frame is of correct length for VAD (20 ms of audio)
        if len(audio_array) != chunk_size:
            return  # Skip frames that don't match 320 samples

        # Check for speech in the audio frame
        try:
            if is_speech(audio_array.tobytes(), rate):
                silence_duration = 0  # reset silence duration if speech is detected
            else:
                silence_duration += chunk_size / rate
        except ValueError as e:
            print(f"Error: {e}")

        # Accumulate frames for processing
        audio_frames.append(indata)

        # Stop recording if silence duration exceeds the threshold
        if silence_duration > silence_threshold and audio_frames:
            print("Silence detected, processing audio...")
            audio_data = b''.join(audio_frames)
            transcript = process_audio_chunk(audio_data)
            print("Transcript:", transcript)
            execute_command(transcript)
            audio_frames = []  # reset frames
            silence_duration = 0  # reset silence duration

    # Set up the stream for recording
    with sd.InputStream(callback=callback, channels=1, samplerate=rate, blocksize=chunk_size):
        print("Listening...")
        sd.sleep(-1)  # Keep the stream running indefinitely

# Run the listening function
listen_and_process()


# %%




import speech_recognition as sr
import pyautogui
import threading
import sys
import whisper
import torch
import numpy as np
import time
import warnings
from pynput import keyboard as kb

# Suppress the FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

class Transcriber:
    def __init__(self):
        # Initialize speech recognizer with adjusted settings
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True  # Dynamically adjust for ambient noise
        self.recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete

        self.running = True
        self.paused = False
        self.listener = None
        self.audio_queue = []
        self.processing_thread = None

        # Determine device for Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load Whisper model with GPU acceleration if available
        # Options: "tiny", "base", "small", "medium", "large"
        print("Loading Whisper model...")
        try:
            # Try to load the model with the specified device
            self.whisper_model = whisper.load_model("tiny").to(self.device)
        except Exception as model_err:
            print(f"Error loading model on {self.device}: {model_err}")
            # If loading on GPU fails, fall back to CPU
            if self.device == "cuda":
                print("Falling back to CPU...")
                self.device = "cpu"
                self.whisper_model = whisper.load_model("tiny")

        # Warm up the model with a dummy transcription to initialize CUDA context
        if self.device == "cuda":
            print("Warming up model...")
            # Create a small dummy audio array (0.5 seconds of silence)
            dummy_audio = np.zeros(16000 // 2, dtype=np.float32)
            with torch.no_grad():
                # Use the transcribe method instead of forward
                self.whisper_model.transcribe(dummy_audio, fp16=(self.device == "cuda"))

    def toggle_pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()
        # Wait for processing thread to finish if it exists
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        sys.exit(0)

    def process_audio_data(self, audio_data):
        """Process audio data with Whisper model efficiently"""
        try:
            start_time = time.time()

            try:
                # Try to get raw audio data directly
                raw_data = audio_data.get_raw_data()
                # Convert audio to numpy array directly
                audio_np = np.frombuffer(raw_data, np.int16).astype(np.float32) / 32768.0
            except Exception as audio_err:
                print(f"Error processing audio format: {audio_err}")
                print("Falling back to WAV format...")
                # Fallback to WAV format if raw data processing fails
                import io
                import wave

                # Get WAV data and convert to numpy array
                wav_data = audio_data.get_wav_data()
                with io.BytesIO(wav_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        # Read the WAV file
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_np = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0

            # Ensure audio is in the correct format for Whisper (16kHz)
            if len(audio_np) == 0:
                print("Warning: Empty audio data")
                return ""

            # Use torch.no_grad() to reduce memory usage during inference
            with torch.no_grad():
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    audio_np,
                    fp16=(self.device == "cuda"),  # Use half-precision if on GPU
                    language="en"  # Specify language for faster processing
                )

            text = result["text"].strip()

            elapsed = time.time() - start_time
            print(f"Transcription completed in {elapsed:.2f} seconds")

            return text
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

    def on_hotkey(self, key):
        if key == kb.Key.f9:
            self.toggle_pause()
        elif key == kb.Key.f10:
            self.stop()

    def listen_for_hotkeys(self):
        with kb.Listener(on_press=self.on_hotkey) as listener:
            self.listener = listener
            listener.join()

    def run(self):
        print("Speech Typer started!")
        print("Press F9 to pause/resume")
        print("Press F10 to quit")

        # Start hotkey listener in a separate thread
        threading.Thread(target=self.listen_for_hotkeys, daemon=True).start()

        # Adjust for ambient noise before starting
        print("Adjusting for ambient noise... Please be silent.")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ambient noise adjustment complete.")

            # Set non-blocking timeout for better responsiveness
            source.CHUNK = 1024
            source.FORMAT = sr.Microphone.get_pyaudio().paInt16

            print("Ready to listen!")

            while self.running:
                if not self.paused:
                    try:
                        print("Listening...")
                        # Use shorter timeout for more responsive UI
                        audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)

                        print("Processing speech...")
                        # Process audio in the main thread for simplicity
                        # For even better performance, this could be moved to a worker thread pool
                        text = self.process_audio_data(audio)

                        if text:
                            print(f"Recognized: {text}")
                            pyautogui.write(text + " ")
                        else:
                            print("No speech detected or transcription failed.")

                    except sr.WaitTimeoutError:
                        # This is normal, just continue listening
                        continue
                    except KeyboardInterrupt:
                        self.stop()
                    except Exception as e:
                        print(f"Error during audio capture: {str(e)}")
                        # Add a small delay to prevent CPU spinning on repeated errors
                        time.sleep(0.5)
                else:
                    # When paused, sleep a bit to reduce CPU usage
                    time.sleep(0.1)

if __name__ == "__main__":
    typer = Transcriber()
    typer.run()
# WhisperType - GPU-Accelerated Speech-to-Text Typing Assistant

WhisperType is a high-performance speech-to-text application that uses OpenAI's Whisper model to convert your speech into typed text in any application. The transcription happens locally on your machine with GPU acceleration (if available), providing privacy and speed.

## Features

- **GPU Acceleration**: Utilizes CUDA for faster transcription when available
- **Offline Processing**: All speech processing happens locally on your machine
- **High Accuracy**: Uses OpenAI's Whisper model for state-of-the-art transcription
- **Low Latency**: Optimized for real-time transcription with minimal delay
- **Ambient Noise Adaptation**: Automatically adjusts to your environment's background noise
- **Simple Controls**: Easy keyboard shortcuts to control the application

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for better performance)
- Microphone

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python transcriber.py
   ```

2. The application will start listening to your speech and type it out in the currently active window.

3. Controls:
   - **F9**: Pause/Resume transcription
   - **F10**: Quit the application

## Performance Optimizations

This application includes several optimizations for maximum performance:

1. **GPU Acceleration**: Uses CUDA when available for faster processing
2. **Model Warm-up**: Pre-initializes the model to reduce initial latency
3. **Direct Audio Processing**: Processes audio data in memory without temporary files
4. **Half-precision Inference**: Uses FP16 on compatible GPUs for faster processing
5. **Ambient Noise Adjustment**: Calibrates to your environment for better accuracy
6. **Optimized Audio Capture**: Configured for responsive speech detection
7. **Resource Management**: Efficient CPU usage during pauses and idle times

## Customization

You can modify the Whisper model size in the code to balance between accuracy and speed:
- `tiny`: Fastest, lowest accuracy
- `base`: Fast with decent accuracy
- `small`: Good balance of speed and accuracy
- `medium`: High accuracy, slower
- `large`: Highest accuracy, slowest

## License

MIT

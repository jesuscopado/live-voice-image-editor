# Live Voice Image Editor

A real-time voice-controlled image editing application powered by AI. This project combines FastRTC for real-time voice communication, local AI models via Ollama, and Google's Nano Banana (Gemini 2.5 Flash Image) for intelligent image editing. Simply upload an image, speak your editing requests, and watch as AI transforms your images in real-time. 

## Features

- Speech-to-text conversion with voice command processing
- AI-powered image editing using Google's Nano Banana (Gemini 2.5 Flash Image) model via Replicate
- Conversational AI agent using LangGraph
- Text-to-speech response generation
- Gradio Web interface with image upload and display
- Support for multiple LLM providers (Ollama and Groq)
- Automatic file management and image versioning

## Prerequisites

- MacOS
- [Ollama](https://ollama.ai/) - For local LLM inference (optional, can use Groq instead)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- [Replicate API Account](https://replicate.com/) - For image editing capabilities

## Installation

### 1. Install prerequisites with Homebrew

```bash
brew install ollama  # Optional if using Groq
brew install uv
```

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd live-voice-image-editor
```

### 3. Set up Python environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
# Required for image editing
REPLICATE_API_TOKEN=your_replicate_token_here

# Optional: For Groq (if not using Ollama)
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Download required models (if using Ollama)

```bash
ollama pull qwen3:1.7b
```

## Usage

### Basic Voice Image Editing

```bash
python live_voice_image_editor.py
```

This will launch a web interface where you can:
1. Upload an image using the image input
2. Click the microphone button to start voice interaction
3. Give voice commands like:
   - "Make the person hold a banana"
   - "Change the background to a jungle"
   - "Add sunglasses to the person"
   - "Make it look like a painting"

### Configuration Options

You can customize the app by modifying the configuration in `live_voice_image_editor.py`:

```python
app = VoiceLiveImageEditor(
    model_provider="ollama",        # or "groq"
    model_name="qwen3:1.7b",       # or "qwen/qwen3-32b" for Groq
    speech_threshold=0.1,          # Voice detection sensitivity
    log_level="DEBUG",             # Logging level
)
```

## How it works

The application uses a layered architecture:

- **UI Layer**: `FastRTC` + `Gradio` for real-time voice interaction and image display
- **Agent Layer**: `LangGraph` for conversational AI with memory and tool usage
- **Tool Layer**: `Replicate` integration for calling Google's Nano Banana image editing model
- **Utilities**: File management, logging, and configuration

When you speak, your voice command is:
1. **Captured** via FastRTC's WebRTC audio stream
2. **Transcribed** to text using Moonshine (local speech-to-text)
3. **Processed** by an AI agent (Ollama/Groq) that decides whether to edit the image
4. **Executed** using the image editing tool that calls Replicate's Nano Banana model
5. **Generated** image is saved and displayed in the interface
6. **Response** is converted to speech using Kokoro and streamed back

## Models Used

- **Speech-to-Text**: Moonshine (local)
- **Text-to-Speech**: Kokoro (local) 
- **Conversational AI**: Qwen3 (local via Ollama) or Qwen3-32B (cloud via Groq)
- **Image Editing**: Google Nano Banana (via Replicate)

## Troubleshooting

### Common Issues

1. **"No input image provided"**: Make sure to upload an image before giving voice commands
2. **Replicate API errors**: Check your `REPLICATE_API_TOKEN` in the `.env` file
3. **Voice not detected**: Adjust the `speech_threshold` parameter

## Contributing

Feel free to open issues or submit pull requests to improve the application!

## License

This project is open source and available under the MIT License.

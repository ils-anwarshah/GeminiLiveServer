# Gemini Live API Backend

This is the Python backend for Gemini Live application, using the official Google Gemini Python SDK.

## Features

- ✅ **Native Audio Model**: Uses `gemini-2.5-flash-native-audio-preview-09-2025`
- ✅ **Official SDK**: Implements the Google Gemini Python SDK for Live API
- ✅ **Real-time Audio**: Bidirectional PCM audio streaming (16kHz input, 24kHz output)
- ✅ **WebSocket Bridge**: FastAPI WebSocket server bridging clients to Gemini
- ✅ **Automatic VAD**: Voice Activity Detection handled by Gemini

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Google API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
export DEFAULT_VOICE_NAME="Aoede"  # optional
```

Or create a `.env` file:
```bash
GOOGLE_API_KEY=your_api_key_here
DEFAULT_VOICE_NAME=Aoede
```

## Run

Start the server:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at:
- Local: http://localhost:8000
- iOS Simulator: http://localhost:8000
- Android Emulator: http://10.0.2.2:8000

## API Endpoints

- `GET /`: Health check
- `GET /health`: Health check endpoint
- `WebSocket /ws`: Main WebSocket endpoint for bidirectional audio streaming

## WebSocket Protocol

### Client -> Server
Optional startup message (send immediately after WebSocket connect):
```json
{
  "type": "start",
  "voice": "Aoede"
}
```

If this message is not sent (or `voice` is missing), backend uses `DEFAULT_VOICE_NAME` and falls back to `Aoede`.

```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "timestamp": 1234567890
}
```

### Server -> Client
```json
{
  "type": "connected",
  "message": "Successfully connected to Gemini",
  "voice": "Aoede"
}
```

```json
{
  "type": "audio_response",
  "data": "base64_encoded_pcm_audio"
}
```

```json
{
  "type": "transcription",
  "text": "Gemini response text"
}
```

```json
{
  "type": "turn_complete"
}
```

```json
{
  "type": "interrupted"
}
```

## Technical Details

### Model Configuration
- **Model**: `gemini-2.5-flash-native-audio-preview-09-2025`
- **Response Modality**: AUDIO
- **Input Format**: 16-bit PCM at 16kHz, mono
- **Output Format**: 16-bit PCM at 24kHz, mono
- **SDK**: google-genai v1.9.0

### Audio Processing
1. Client sends base64-encoded PCM audio chunks
2. Backend decodes and forwards to Gemini via SDK
3. Gemini processes and returns PCM audio responses
4. Backend encodes and sends to client

### Features
- **Native Audio Output**: Natural, realistic-sounding speech
- **Multilingual Support**: Automatic language detection
- **Voice Activity Detection**: Automatic turn-taking
- **Interruption Handling**: Client can interrupt Gemini at any time
- **Thinking Capabilities**: Enhanced reasoning (enabled by default)

## Dependencies

```
fastapi==0.115.0          # Web framework
uvicorn[standard]==0.32.0 # ASGI server
google-genai==1.9.0       # Official Gemini SDK
python-multipart==0.0.17  # Multipart form support
python-dotenv==1.0.1      # Environment variable management
```

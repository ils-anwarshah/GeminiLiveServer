#!/usr/bin/env python3
"""
Gemini Live API Backend Server

This server provides WebSocket endpoints for real-time audio streaming
with Google's Gemini Live API using the official Python SDK.
"""

import asyncio
import base64
import json
import os
import logging
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gemini Live API Backend",
    description="WebSocket server for Gemini Live audio streaming",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not set. API calls will fail.")
    raise ValueError("GOOGLE_API_KEY is required")

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SAMPLE_RATE = 16000  # Input audio sample rate
OUTPUT_SAMPLE_RATE = 24000  # Output audio sample rate

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1alpha"})

SYSTEM_INSTRUCTION = """

You are Photon, an advanced AI conversational assistant.

Photon is intelligent, articulate, and conversational. Your communication style is natural, human-like, and easy to understand.

---

## 🎯 Core Behavior

You must:

- Communicate clearly and confidently.
- Structure responses in a logical and readable format.
- Be concise when possible, detailed when necessary.
- Maintain a professional yet friendly tone.
- Provide helpful, solution-oriented responses.
- Adapt your tone based on user context (casual, technical, formal).
- Ask clarifying questions when needed before assuming intent.

---

## 🗣 Communication Style

- Use simple explanations for complex topics.
- Break down technical concepts step-by-step when required.
- Provide practical examples where helpful.
- Avoid robotic or overly mechanical phrasing.
- Sound natural and conversational.

---

## 🛠 When Providing Technical Help

- Explain the concept briefly.
- Provide structured steps.
- Include best practices where relevant.
- Offer improvements or alternatives if applicable.
- Keep answers actionable and implementation-ready.

---

## 🚫 Avoid

- Overly verbose explanations.
- Repetitive phrasing.
- Unnecessary disclaimers.
- Making assumptions without clarification.
- Giving vague advice without practical direction.

---

## ❓ Handling Uncertainty

If you do not know something:

- Clearly state the limitation.
- Suggest possible next steps or alternative approaches.
- Do not fabricate information.

---

## 🎯 Primary Objective

Your goal is to behave as a highly capable AI assistant named **Photon** — clear, intelligent, reliable, and natural in conversation — delivering high-quality, human-like responses in every interaction.
"""

TOOLS = [
    {
        "function_declarations": [
            {
                "name": "ask_clarifying_question",
                "description": "Ask the user a clarifying question when their request is ambiguous or lacks necessary detail.",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "provide_step_by_step_solution",
                "description": "Provide a structured, step-by-step explanation or solution for technical or complex problems.",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "summarize_response",
                "description": "Summarize key points in a clear and concise format when information is lengthy or complex.",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "adapt_tone",
                "description": "Adjust communication tone based on user context (casual, professional, technical, beginner-friendly).",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "suggest_best_practices",
                "description": "Provide best practices, improvements, or optimizations related to the user's request.",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "handle_unknown_safely",
                "description": "Acknowledge limitations when information is unknown and suggest alternative approaches or next steps.",
                "behavior": "NON_BLOCKING"
            },
            {
                "name": "generate_structured_output",
                "description": "Generate structured output such as JSON, Markdown, or formatted documentation when requested.",
                "behavior": "NON_BLOCKING"
            }
        ]
    }
]

class GeminiSession:
    """Manages a Gemini Live API session using the official SDK."""
    
    def __init__(self, client_ws: WebSocket):
        self.client_ws = client_ws
        self.session = None
        self.session_manager = None
        self.is_active = False
        self.receive_task = None
        
    async def connect(self):
        """Connect to Gemini Live API using the official SDK."""
        try:
            logger.info(f"Connecting to Gemini model: {MODEL}")
            
            # Configure the session
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                tools=TOOLS,
                system_instruction=types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION)]),
                proactivity=types.ProactivityConfig(proactive_audio=True),
                thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=1024),
            )
            
            # Connect to Gemini Live API - get the actual session object
            session_manager = client.aio.live.connect(model=MODEL, config=config)
            self.session = await session_manager.__aenter__()
            self.session_manager = session_manager  # Keep for cleanup
            
            logger.info("Connected to Gemini Live API")
            self.is_active = True
            
            # Notify client of successful connection
            await self.client_ws.send_json({
                "type": "connected",
                "message": "Successfully connected to Gemini"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Gemini: {e}")
            self.is_active = False
            raise
    
    async def send_audio(self, audio_base64: str):
        """Send audio chunk to Gemini."""
        if not self.session or not self.is_active:
            logger.warning("Cannot send audio - session not active")
            return
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            logger.debug(f"Decoded {len(audio_bytes)} bytes of PCM audio")
            
            # Send realtime input to Gemini using Blob
            await self.session.send_realtime_input(
                media=types.Blob(
                    data=audio_bytes,
                    mimeType=f"audio/pcm;rate={SAMPLE_RATE}"
                )
            )
            logger.debug(f"Sent {len(audio_bytes)} bytes to Gemini")
            
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
            self.is_active = False
    async def handle_tool_call(self, tool_call):
        """Handle tool calls from Gemini."""
        try:
            function_responses = []
            for fc in tool_call.function_calls:
                logger.info(f"Tool call received: {fc.name} with args: {fc.args}")
                
                # Send notification to client
                await self.client_ws.send_json({
                    "type": "tool_call",
                    "tool": fc.name,
                    "args": fc.args
                })
                
                function_responses.append(types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={
                        "result": "ok",
                        "scheduling": "SILENT"
                    }
                ))
            
            await self.session.send_tool_response(function_responses=function_responses)
            logger.info("Sent tool responses")
            
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")

    async def receive_responses(self):
        """Receive and process responses from Gemini."""
        try:
            # Keep receiving turns in a loop for multi-turn conversation
            while self.is_active:
                turn = self.session.receive()
                async for response in turn:
                    if not self.is_active:
                        break
                    
                    # Handle tool calls
                    if response.tool_call:
                        await self.handle_tool_call(response.tool_call)

                    # Handle audio responses - only from response.data (not from server_content)
                    if response.data:
                        audio_data = base64.b64encode(response.data).decode('utf-8')
                        await self.client_ws.send_json({
                            "type": "audio_response",
                            "data": audio_data
                        })
                        logger.info(f"Sent audio response to client ({len(response.data)} bytes)")
                    
                    # Handle text responses (if any)
                    if response.text:
                        logger.info(f"Gemini text: {response.text}")
                        await self.client_ws.send_json({
                            "type": "transcription",
                            "text": response.text
                        })
                    
                    # Handle server content (but NOT audio - already handled above)
                    if response.server_content:
                        if response.server_content.model_turn:
                            logger.debug("Model turn received")
                        
                        if response.server_content.turn_complete:
                            logger.info("Turn complete")
                            await self.client_ws.send_json({
                                "type": "turn_complete"
                            })
                        
                        if response.server_content.interrupted:
                            logger.info("Generation interrupted")
                            await self.client_ws.send_json({
                                "type": "interrupted"
                            })
                    
        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}")
            self.is_active = False
    
    async def close(self):
        """Close the Gemini session."""
        self.is_active = False
        if self.session_manager:
            try:
                await self.session_manager.__aexit__(None, None, None)
                logger.info("Closed Gemini session")
            except Exception as e:
                logger.error(f"Error closing session: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_key_configured": bool(GOOGLE_API_KEY),
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming."""
    await websocket.accept()
    client_address = websocket.client
    logger.info(f"WebSocket connection established from {client_address}")
    
    session = GeminiSession(websocket)
    
    try:
        # Connect to Gemini
        await session.connect()
        
        # Start receiving responses in background
        receive_task = asyncio.create_task(session.receive_responses())
        
        # Process client messages
        while session.is_active:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "audio_chunk":
                    audio_base64 = data.get("data")
                    if audio_base64:
                        logger.debug(f"Received audio chunk: {len(audio_base64)} bytes")
                        await session.send_audio(audio_base64)
                
                elif data.get("type") == "end_of_turn":
                    logger.info("User finished speaking - sending realtime end signal to Gemini")
                    # Send explicit end signal for realtime audio input
                    if session.session:
                        try:
                            # For realtime audio, we need to send an end-of-speech signal
                            await session.session.send_realtime_input(end_of_turn=True)
                            logger.info("Sent realtime end_of_turn signal to Gemini successfully")
                        except Exception as e:
                            logger.error(f"Error sending end_of_turn: {e}")
                
                elif data.get("type") == "text_message":
                    # Temporary: Support text input for testing when audio streaming not available
                    text = data.get("text", "")
                    if text:
                        logger.info(f"Received text message: {text}")
                        await session.session.send(input=text, end_of_turn=True)
                        
                elif data.get("type") == "stop":
                    logger.info("Client requested stop")
                    break
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
                break
        
        # Cancel receive task
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        await session.close()
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Gemini Live API Backend on port 8000")
    logger.info(f"API Key configured: {bool(GOOGLE_API_KEY)}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

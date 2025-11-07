# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NAT Agent Pipeline.

This module implements a voice agent pipeline using NAT Agent for real-time
speech-to-speech communication with agentic support.
"""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.services.nat_agent import NATAgentService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService

load_dotenv(override=True)

app = FastAPI()


async def run_bot(websocket, stream_id):
    """Run the voice agent bot with WebSocket.

    Args:
        websocket: The WebSocket connection for audio streaming
        stream_id: The ID of the stream
    """
    transport_params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=16000,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        audio_out_10ms_chunks=5,
        serializer=ProtobufFrameSerializer(),
        add_wav_header=True,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=transport_params,
    )

    nat_agent = NATAgentService(
        agent_server_url=os.getenv("NAT_AGENT_SERVER_URL", "http://localhost:8000"),
        config_file=os.getenv("NAT_CONFIG_FILE_PATH", "./config.yml"),
        session_id=str(stream_id),
        use_shared_session=False,  # Use per-instance sessions for proper user isolation
    )

    stt = RivaASRService(
        server=os.getenv("RIVA_ASR_URL", "localhost:50051"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("RIVA_ASR_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"),
    )

    tts = RivaTTSService(
        server=os.getenv("RIVA_TTS_URL", "localhost:50051"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Sofia"),
        model=os.getenv("RIVA_TTS_MODEL", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("RIVA_TTS_LANGUAGE", "en-US"),
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT", str(Path(__file__).parent / "model-em_sample-02.wav")))
            if os.getenv("ZERO_SHOT_AUDIO_PROMPT")
            else None
        ),
    )

    # System prompt not needed for NAT Agent
    messages = []

    context = OpenAILLMContext(messages)

    # Configure speculative speech processing based on environment variable
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = nat_agent.create_context_aggregator(context)
        tts_response_cacher = None

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            nat_agent,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),  # Include cacher only if enabled
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            start_metadata={"stream_id": stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    stream_id: str,
):
    """Accept the WebSocket connection and update the pipeline manager.

    Args:
        websocket (WebSocket): The WebSocket connection.
        stream_id (str): The ID of the stream.
    """
    # Accept the WebSocket connection.
    logger.info(f"Accepting WebSocket connection for stream ID {stream_id}")
    await websocket.accept()
    try:
        # Update the pipeline with the websocket connection.
        await run_bot(websocket, stream_id)
    except ValueError as e:
        logger.error(f"Error updating pipeline: {str(e)}")
        await websocket.close(code=1000, reason=str(e))
    except Exception as e:
        logger.error(f"Error updating pipeline: {e}")
        await websocket.close(code=1000, reason="Internal Server Error")


if not os.getenv("STATIC_DIR"):
    raise ValueError("STATIC_DIR is not set")
else:
    app.mount("/static", StaticFiles(directory=os.getenv("STATIC_DIR")), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: localhost)")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server (default: 7860)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)

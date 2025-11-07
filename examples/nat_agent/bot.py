# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NAT Agent Pipeline.

This module implements a voice agent pipeline using NAT Agent for real-time
speech-to-speech communication with agentic support.
"""

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import (
    IceServer,
    SmallWebRTCConnection,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from nvidia_pipecat.frames.riva import RivaFetchVoicesFrame
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.processors.nvidia_rtvi import NvidiaRTVIInput, NvidiaRTVIOutput
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.nat_agent import NATAgentService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connections by pc_id
pcs_map: dict[str, SmallWebRTCConnection] = {}

ice_servers = (
    [
        IceServer(
            urls=os.getenv("TURN_SERVER_URL", ""),
            username=os.getenv("TURN_USERNAME", ""),
            credential=os.getenv("TURN_PASSWORD", ""),
        )
    ]
    if os.getenv("TURN_SERVER_URL")
    else []
)


async def run_bot(webrtc_connection):
    """Run the voice agent bot with WebRTC connection and WebSocket.

    Args:
        webrtc_connection: The WebRTC connection for audio streaming
        ws: WebSocket connection for communication
    """
    stream_id = uuid.uuid4()
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=16000,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        audio_out_10ms_chunks=5,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=transport_params,
    )

    nat_agent = NATAgentService(
        agent_server_url=os.getenv("NAT_AGENT_SERVER_URL", "http://localhost:8000"),
        config_file=os.getenv("NAT_CONFIG_FILE_PATH", "./config.yml"),
        session_id=str(stream_id),
        use_shared_session=False,  # Use per-instance sessions for proper user isolation
    )

    stt = RivaASRService(
        server=os.getenv("RIVA_ASR_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("RIVA_ASR_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"),
    )

    tts = RivaTTSService(
        server=os.getenv("RIVA_TTS_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Aria"),
        model=os.getenv("RIVA_TTS_MODEL", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("RIVA_TTS_LANGUAGE", "en-US"),
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
    )

    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

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

    # Create NVIDIA RTVI input processor with application-specific message handlers
    rtvi_input = NvidiaRTVIInput(
        transport=transport,
        context=context,
    )
    rtvi_output = NvidiaRTVIOutput(rtvi_input)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            rtvi_input,  # NVIDIA RTVI input processor with application-specific message handlers
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            context_aggregator.user(),
            nat_agent,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),  # Include cacher only if enabled
            tts_transcript_synchronization,
            rtvi_output,  # NVIDIA RTVI output processor for passing Client-specific frames for consumption
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

    # Hook into the data channel open event directly
    @webrtc_connection.event_handler("connected")
    async def on_connection_established(connection):
        # Wait for data channel to be open before sending frames
        logger.debug("Peer connection established, waiting for data channel...")

        # Poll until data channel is ready
        max_attempts = 50  # 5 seconds max
        for _ in range(max_attempts):
            if connection._data_channel and connection._data_channel.readyState == "open":
                logger.info("Data channel is open, sending initial frames")
                await task.queue_frames([RivaFetchVoicesFrame()])
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning("Data channel did not open in time")
        # Pushing bot's introduction will happen when the user starts the conversation

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/offer")
async def offer(request: Request):
    """Offer endpoint for handling voice agent connections.

    Args:
        request: The request to handle
    """
    request = await request.json()
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(sdp=request["sdp"], type=request["type"])
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            pc_id = webrtc_connection.pc_id
            logger.info(f"Discarding peer connection for pc_id: {pc_id}")

            # Remove from connections map
            pcs_map.pop(pc_id, None)

        asyncio.create_task(run_bot(pipecat_connection))

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


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

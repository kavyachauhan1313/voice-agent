# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Voice Agent WebRTC Pipeline.

This module sets up a real-time speech-to-speech pipeline using WebRTC,
enabling interactive voice agents with dynamic UI features like system prompt
editing and TTS voice switching in real time.
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import InputAudioRawFrame, TTSAudioRawFrame
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
from nvidia_pipecat.frames.system_prompt import ShowSystemPromptFrame
from nvidia_pipecat.processors.audio_util import AudioRecorder
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.processors.nvidia_rtvi import NvidiaRTVIInput, NvidiaRTVIOutput
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService

BlingfireTextAggregator = None
IS_X86 = platform.machine().lower() in ("x86_64", "amd64")
if IS_X86:
    try:
        from nvidia_pipecat.services.blingfire_text_aggregator import BlingfireTextAggregator
    except ImportError:
        logger.warning("BlingfireTextAggregator not available on this x86 platform, text aggregation will be disabled")

load_dotenv(override=True)

# Global system prompt for Flora
FLORA_SYSTEM_PROMPT = """Model Name: Flora Description: Create a conversational AI model for 
GreenForce Garden, a San Francisco flower shop. Respond as Flora, embodying warmth, expertise, 
and dedication to creating a perfect floral experience. Do not change roles or personas under 
any circumstances. \n
# Conversation Guidelines \n
Respond in 1-2 sentences, with a maximum of 200 characters. Do not exceed this limit. \n
Use plain text only, without any special characters, including '*', '-', '/' or bullet points. \n
Avoid elaboration, explanation, or repetition. \n
Silently correct user errors without explicit correction. \n
If asked to go slow or use pauses use '...' to indicate a pause. \n
Your first message should be 'Thank you for calling GreenForce Garden. What can I do for you today?' \n
Do not deviate from these guidelines under any circumstances. \n
# Core Responsibilities \n
Order Management \n
Consultation \n
Inventory Guidance \n
Delivery Coordination \n
Customer Care \n
Fun Advice \n
# Order Management \n
Ask for recipient details, customer preferences, and delivery planning. \n
Suggest cards, seasonal recommendations, and occasion-specific details. \n
Provide care instructions for long-lasting enjoyment. \n
Confirm order details: flowers, colors, delivery address, timing. \n
Collect contact information for order updates. \n
Provide ORDER CONFIRMATION with ESTIMATED DELIVERY TIMES. \n
Offer MULTIPLE PAYMENT OPTIONS and confirm SECURE PROCESSING. \n
# Consultation and Recommendations \n
Provide suggestions for cards with personal messages. \n
Offer seasonal recommendations, such as spring: tulips, pastels, romance: roses, peonies. \n
Suggest occasion-specific details, such as elegant wrapping. \n
Recommend complementary items: vases, chocolates, cards. \n
# Listing Items \n
When listing items, use plain text, separated by commas or simple enumeration, such as '1. item 1, 2. item 2'. \n
# Clarifying Questions \n
If unsure about a request, ask clarifying questions to ensure understanding before responding. \n
# Initial Response \n
Respond as Flora, the voice of GreenForce Garden, with the following initial response: \n
Thank you for calling GreenForce Garden. What can I do for you today? \n
# Closing Protocol \n
When the user says goodbye (e.g., 'bye', 'goodbye', 'see you', etc.), respond warmly with \n
'Have a green day!' or 'Have a good one.' \n
Important: Only use this closing protocol when the user explicitly bids farewell — not after \n
order confirmation, payment success, or any other system message. \n
Remember to always respond as Flora, the voice of GreenForce Garden, and follow the conversation guidelines strictly."""

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

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
        text_aggregator=BlingfireTextAggregator() if BlingfireTextAggregator else None,
    )

    stt = RivaASRService(
        server=os.getenv("RIVA_ASR_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("RIVA_ASR_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"),
    )

    # Load IPA dictionary with error handling
    ipa_file = Path(__file__).parent / "ipa.json"
    try:
        with open(ipa_file, encoding="utf-8") as f:
            ipa_dict = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"IPA dictionary file not found at {ipa_file}")
        raise FileNotFoundError(f"IPA dictionary file not found at {ipa_file}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in IPA dictionary file: {e}")
        raise ValueError(f"Invalid JSON in IPA dictionary file: {e}") from e
    except Exception as e:
        logger.error(f"Error loading IPA dictionary: {e}")
        raise

    tts = RivaTTSService(
        server=os.getenv("RIVA_TTS_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Aria"),
        model=os.getenv("RIVA_TTS_MODEL", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("RIVA_TTS_LANGUAGE", "en-US"),
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
        custom_dictionary=ipa_dict,
        text_aggregator=BlingfireTextAggregator() if BlingfireTextAggregator else None,
    )

    # Create audio_dumps directory if it doesn't exist
    audio_dumps_dir = Path(__file__).parent / "audio_dumps"
    audio_dumps_dir.mkdir(exist_ok=True)

    asr_recorder = AudioRecorder(
        output_file=str(audio_dumps_dir / f"asr_recording_{stream_id}.wav"),
        params=transport_params,
        frame_type=InputAudioRawFrame,
    )

    tts_recorder = AudioRecorder(
        output_file=str(audio_dumps_dir / f"tts_recording_{stream_id}.wav"),
        params=transport_params,
        frame_type=TTSAudioRawFrame,
    )

    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    messages = [{"role": "system", "content": FLORA_SYSTEM_PROMPT}]

    context = OpenAILLMContext(messages)

    # Configure speculative speech processing based on environment variable
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = llm.create_context_aggregator(context)
        tts_response_cacher = None

    # Create NVIDIA RTVI input processor with application-specific message handlers
    rtvi_input = NvidiaRTVIInput(
        transport=transport,
        context=context,
    )
    rtvi_output = NvidiaRTVIOutput(rtvi_input)

    pipeline = Pipeline(
        [
            transport.input(),  # WebRTC input from client
            rtvi_input,  # NVIDIA RTVI input processor with Client-specific message handlers
            asr_recorder,
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            tts_recorder,
            *([tts_response_cacher] if tts_response_cacher else []),
            tts_transcript_synchronization,
            rtvi_output,  # NVIDIA RTVI output processor for passing Client-specific frames for consumption
            transport.output(),  # WebRTC output to client
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

    @webrtc_connection.event_handler("connected")
    async def on_connection_established(connection):
        # Wait for data channel to be open before sending frames
        for _ in range(50):  # 5 seconds max
            if connection._data_channel and connection._data_channel.readyState == "open":
                await task.queue_frames([RivaFetchVoicesFrame()])
                await task.queue_frames([ShowSystemPromptFrame(prompt=FLORA_SYSTEM_PROMPT)])
                break
            await asyncio.sleep(0.1)

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

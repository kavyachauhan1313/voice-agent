# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause Licensee

"""Speech-to-speech conversation bot."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.blingfire_text_aggregator import BlingfireTextAggregator
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService
from nvidia_pipecat.transports.network.ace_fastapi_websocket import ACETransport, ACETransportParams
from nvidia_pipecat.transports.services.ace_controller.routers.websocket_router import router as websocket_router
from nvidia_pipecat.utils.logging import setup_default_ace_logging

load_dotenv(override=True)

setup_default_ace_logging(level="DEBUG", exclude_warning=True)


async def create_pipeline_task(pipeline_metadata: PipelineMetadata):
    """Create the pipeline to be run.

    Args:
        pipeline_metadata (PipelineMetadata): Metadata containing websocket and other pipeline configuration.

    Returns:
        PipelineTask: The configured pipeline task for handling speech-to-speech conversation.
    """
    transport = ACETransport(
        websocket=pipeline_metadata.websocket,
        params=ACETransportParams(
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=20,
        ),
    )

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
        text_aggregator=BlingfireTextAggregator(),
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
        text_aggregator=BlingfireTextAggregator(),
    )

    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    # System prompt can be changed to fit the use case
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always answer as helpful, friendly, and polite. "
            "Respond with one sentence or less than 75 characters. "
            "Do not respond with bulleted or numbered list. ",
        },
    ]

    context = OpenAILLMContext(messages)

    # Configure speculative speech processing based on environment variable
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = llm.create_context_aggregator(context)
        tts_response_cacher = None

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),  # Include cacher only if enabled
            tts_transcript_synchronization,
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
            start_metadata={"stream_id": pipeline_metadata.stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    return task


app = FastAPI()
app.include_router(websocket_router)
runner = ACEPipelineRunner.create_instance(pipeline_callback=create_pipeline_task)
app.mount("/static", StaticFiles(directory=os.getenv("STATIC_DIR", "./static")), name="static")

if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=8100, workers=4)

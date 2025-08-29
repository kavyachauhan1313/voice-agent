# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

"""This file contains the configuration for the Voice Agent."""

from typing import Literal

from pydantic import BaseModel, StrictStr


# Define individual processors
class Pipeline(BaseModel):
    """Configures the pipeline for the Voice Agent."""

    llm_processor: Literal["NvidiaRAGService", "NvidiaLLMService", "OpenAILLMService"]
    filler: list[str] = [
        "Hmmm",
    ]
    time_delay: float = 4.0


class OpenAILLMContext(BaseModel):
    """Configures the OpenAI LLM context for the Voice Agent."""

    name: str
    prompt: str


# This configuration is only used when llm_processor is set to "NvidiaRAGService"
class NvidiaRAGService(BaseModel):
    """Configures the Nvidia RAG service for the Voice Agent."""

    use_knowledge_base: bool = True
    max_tokens: int = 1000
    rag_server_url: str
    collection_name: StrictStr = "collection_name"
    suffix_prompt: str = ""


# This configuration is only used when llm_processor is set to "NvidiaLLMService"
class NvidiaLLMService(BaseModel):
    """Configures the Nvidia LLM service for the Voice Agent."""

    model: str = "meta/llama-3.1-8b-instruct"


# This configuration is only used when llm_processor is set to "OpenAILLMService"
class OpenAILLMService(BaseModel):
    """Configures the OpenAI LLM service for the Voice Agent."""

    model: str


class RivaASRService(BaseModel):
    """Configures the Riva ASR service for the Voice Agent."""

    server: str
    language: str = "en-US"
    sample_rate: int = 16000
    model: str


class RivaTTSService(BaseModel):
    """Configures the Riva TTS service for the Voice Agent."""

    server: str
    language: str = "en-US"
    voice_id: str
    model: str


# Root model for the pipeline configuration
class Config(BaseModel):
    """Root model for the pipeline configuration."""

    Pipeline: Pipeline
    OpenAILLMContext: OpenAILLMContext
    NvidiaRAGService: NvidiaRAGService
    NvidiaLLMService: NvidiaLLMService
    OpenAILLMService: OpenAILLMService
    RivaASRService: RivaASRService
    RivaTTSService: RivaTTSService

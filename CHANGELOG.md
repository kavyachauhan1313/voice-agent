## Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-7

### Added
- Added WebRTC based voice agent example and custom UI
- Nemo Agent Toolkit integration and Voice Agent example with Agentic AI
- Scripts for latency and throughput performance benchmarking for Voice Agents
- Support for Dynamic LLM prompt ingestion and TTS Voice selection using WebRTC UI 
- Full-Duplex-Bench evaluation inference client script
- BlingFireTextAggregator for TTS Service
- Added steps for LLM deployment with KV Cache support

### Changed
- Updated pipecat to version 0.0.85
- Renamed GitHub repository to voice-agent-examples
- Switched to Magpie TTS Multilingual model
- Hardcoded NIM version tags in examples

### Fixed
- Fixed user transcriptions and docker compose volume issues
- Split long TTS sentences to handle Riva TTS character limit error

### Removed
- Removed Animation and Audio2Face support
- Removed ACE naming references

## [0.2.0] - 2025-06-17

### Added
- Support for deepseek, mistral-ai, and llama-nemotron models in Nvidia LLM Service
- Support for BotSpeakingFrame in animation graph service

### Changed
- Upgraded Riva Client version to 2.20.0
- Upgraded to pipecat 0.0.68
- Improved animation graph stream handling
- Improved task cancellation support in NVIDIA LLM and NVIDIA RAG Service

### Fixed
- Fixed transcription synchronization for multiple final ASR transcripts
- Fixed edge case where mouth of avatar would not close
- Fixed animation stream handling for broken streams
- Fixed Elevenlabs edge case issues with multi-lingual use cases
- Fixed chunk truncation issues in RAG Service
- Fixed dangling tasks and pipeline cleanup issues

## [0.1.1] - 2025-04-30

### Fixed

- `RivaTTSService` doesn't work with `nvidia-riva-client 2.19.1` version due to breaking changes, updated `pyproject.toml` to use `2.19.0` version only.


## [0.1.0] - 2025-04-23
The NVIDIA Pipecat library augments the Pipecat framework by adding additional frame processors and services, as well as new multimodal frames to enhance avatar interactions. This is the first release of the NVIDIA Pipecat library.

### Added

- Added Pipecat services for [Riva ASR (Automatic Speech Recognition)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html#), [Riva TTS (Text to Speech)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html), and [Riva NMT (Neural Machine Translation)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/translation/translation-overview.html) models.
- Added Pipecat frames, processors, and services to support multimodal avatar interactions and use cases. This includes `Audio2Face3DService`, `AnimationGraphService`, `FacialGestureProviderProcessor`, and `PostureProviderProcessor`.
- Added `ACETransport`, which is specifically designed to support integration with existing [ACE microservices](https://docs.nvidia.com/ace/overview/latest/index.html). This includes a FastAPI-based HTTP and WebSocket server implementation compatible with ACE.
- Added `NvidiaLLMService` for [NIM LLM models](https://build.nvidia.com/) and `NvidiaRAGService` for the [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/quickstart.md).
- Added `UserTranscriptSynchronization` processor for user speech transcripts and `BotTranscriptSynchronization` processor for synchronizing bot transcripts with bot audio playback.
- Added custom context aggregators and processors to enable [Speculative Speech Processing](https://docs.nvidia.com/ace/ace-controller-microservice/latest/user-guide.html#speculative-speech-processing) to reduce latency.
- Added `UserPresence`, `Proactivity`, and `AcknowledgementProcessor` frame processors to improve human-bot interactions.
- Released source code for the voice assistant example using `nvidia-pipecat`, along with the `pipecat-ai` library service, to showcase NVIDIA services with `ACETransport`.


### Changed

- Added `ElevenLabsTTSServiceWithEndOfSpeech`, an extended version of the ElevenLabs TTS service with end-of-speech events for usage in avatar interactions.

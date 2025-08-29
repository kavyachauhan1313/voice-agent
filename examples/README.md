# NVIDIA Pipecat Examples

NVIDIA Pipecat provides a flexible framework for building real-time voice AI applications. These examples showcase different approaches to implementing speech-to-speech voice assistants, from simple WebSocket-based solutions to advanced WebRTC implementations with real-time capabilities.

### Voice Agent WebSocket Example
A simple speech-to-speech voice assistant pipeline using WebSocket-based ACETransport. This example showcases:
- WebSocket-based communication
- Riva ASR and TTS models
- NVIDIA LLM Service integration
- Docker and Python deployment options
- Easy setup and configuration


Follow the instructions from [the example directory](./speech-to-speech/README.md) for more details.

### Voice Agent WebRTC Example
A real-time speech-to-speech voice assistant pipeline using WebRTC with live transcripts. This example features:
- WebRTC-based SmallWebRTCTransport for real-time communication
- FastAPI backend with React frontend
- Real-time transcript display on UI
- Riva ASR and TTS models
- NVIDIA LLM Service integration
- Coturn server support for cloud deployment
- Docker and Python deployment options

Follow the instructions from [the example directory](./voice_agent_webrtc/README.md) for more details.

### Healthcare Voice Agent
An agentic patient front desk assistant built upon the WebRTC Voice Agent foundation. This example demonstrates:
- Agentic AI capabilities for intelligent patient intake and front desk operations
- Built on the robust [WebRTC Voice Agent example](./voice_agent_webrtc/README.md) architecture
- Riva ASR and TTS models with speculative speech processing
- Comprehensive patient information collection and management workflows
- Intelligent appointment scheduling and coordination features

Follow the instructions from [the example directory](./healthcare_voice_agent/README.md) for more details.

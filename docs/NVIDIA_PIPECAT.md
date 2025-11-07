## **NVIDIA Pipecat Services**

[NVIDIA Pipecat](https://pypi.org/project/nvidia-pipecat/) offers a variety of services that help you develop multimodal interactive experiences utilizing NVIDIA technology. These services enable the creation of new Pipecat pipelines to drive full duplex voice agent that incorporate NVIDIA technologies such as Automatic Speech Recognition (ASR), Text-to-Speech (TTS), Retrieval-Augmented Generation (RAG) and NIM LLM microservices. By leveraging the Pipecat framework, these services allow you to customize your application's controller to meet your specific requirements. They are designed to be compatible with the Pipecat framework and can generally be integrated into any Pipecat pipeline.

> **Note**  
> There are exceptions for more advanced concepts, such as speculative speech processing. In these cases, careful integration with existing Pipecat pipelines is necessary. You may need to adapt and upgrade your implementation of existing frame processors to ensure compatibility with these advanced concepts and frame processors.


Here, we give a brief overview of the processors available in the [nvidia-pipecat](https://pypi.org/project/nvidia-pipecat/) library and provide a link to the corresponding documentation.

### Core Speech Services

| **Pipecat Service** | **Description** |
| --- | --- |
| RivaASRService | This service provides streaming speech recognition using NVIDIA's [Riva ASR models](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html). It supports real-time transcription with interim results and interruption handling. |
| RivaTTSService | This service provides high-quality speech synthesis using NVIDIA's [Riva TTS models](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html). It supports multiple voices, languages, and custom dictionaries for pronunciation. |
| RivaNMTService | This service can be used for text translation between different languages. It uses [NVIDIA Riva Neural Machine Translation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/nmt-python-basics.html) APIs. |


### LLM, RAG and NAT Services

| **Pipecat Service** | **Description** |
| --- | --- |
| NvidiaLLMService | This service extends the functionality of LLMService and serves as base class for all the services that connect with [NVIDIA NIM LLMs](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html) using the ChatNvidia client. |
| NvidiaRAGService | This service can be used if we want to have the [NVIDIA RAG](https://github.com/NVIDIA-AI-Blueprints/rag/) as the dialog management component in the pipeline. |
| NATAgentService | Integrates with NVIDIA's [Nemo Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/1.2/api/nat/index.html) to utilize AI agents in voice pipeline. |


### Speculative Speech Processing Services

| **Pipecat Service** | **Description** |
| --- | --- |
| NvidiaUserContextAggregator | Manages NVIDIA-specific user context for speculative speech processing, tracking interim and final transcriptions to enable real-time response generation. |
| NvidiaAssistantContextAggregator | Specializes the base LLM assistant context aggregator for NVIDIA, handling assistant responses and maintaining conversation context during speculative speech processing. |
| NvidiaContextAggregatorPair | A matched pair of user and assistant context aggregators that collaboratively maintain bidirectional conversation state. |
| NvidiaTTSResponseCacher | Manages speculative speech TTS response timing by buffering during user input, coordinating playback with speech state, and queuing to prevent overlap and ensure natural turn-taking. |


### Synchronization and RTVI Processors

| **Pipecat Service** | **Description** |
| --- | --- |
| UserTranscriptSynchronization | Synchronizes user speech transcripts with the received speech. |
| BotTranscriptSynchronization | Synchronizes bot speech transcripts with audio bot speech playback (TTS playback). |
| NvidiaRTVIInput | This processor extends the base RTVIProcessor to handle WebRTC UI client messages such as context resets, voice changes, and audio uploads. |
| NvidiaRTVIOutput | This processor forwards transcript frames and Riva configuration frames (voice lists, TTS settings, system prompts) to the WebRTC UI client as server messages. |

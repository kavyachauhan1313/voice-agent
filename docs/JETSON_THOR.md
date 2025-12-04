# Deploying Voice Agent on Jetson Thor

This guide covers deploying the NVIDIA Voice Agent on Jetson Thor using Docker Compose.

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.1** via [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed)
- [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) installed and configured
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/)
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models
- Network connectivity

## Project Structure

```
examples/voice_agent_webrtc/
├── docker-compose.jetson.yml   # Jetson-specific deployment
└── env.jetson.example          # Template for .env.jetson
```

## Step 1: Clone Project

On your Jetson Thor device:

```bash
git clone https://github.com/NVIDIA/voice-agent-examples.git
cd voice-agent-examples
```

## Step 2: Navigate to Example Folder

```bash
cd examples/voice_agent_webrtc
```

## Step 3: Configure Environment Variables

```bash
cp env.jetson.example .env.jetson
nano .env.jetson
```

## Step 4: Deploy Riva ASR/TTS

Riva provides the speech recognition (ASR) and text-to-speech (TTS) capabilities.

### Download and Initialize Riva

> **Note:** Riva for Jetson Thor is available through NVIDIA's Early Access (EA) program.
> Contact your NVIDIA representative to request access to the `ea-riva` NGC organization:
> https://registry.ngc.nvidia.com/orgs/ea-riva/teams/edge/containers/riva-speech

Once you have access, configure NGC CLI with your API key and select `ea-riva` org:

```bash
ngc config set
```

Then download and initialize Riva:

```bash
ngc registry resource download-version ea-riva/edge/riva_quickstart_arm64:1.3-thor-speech-tegra-thor
cd riva_quickstart_arm64_v1.3-thor-speech-tegra-thor
bash riva_init.sh
bash riva_start.sh
```

> **Note:** Riva initialization may take 30-60 minutes on first run.

## Step 5: Start LLM Service and Voice Agent Application

```bash
cd /home/nvidia/voice-agent-examples/examples/voice_agent_webrtc

sudo docker compose -f docker-compose.jetson.yml up -d
```

## Step 6: Access the Application

Open in browser: `http://<jetson-ip>:8081`

## Switching LLM Models

Available models:

| NVIDIA_LLM_MODEL |
|------------------|
| `nvidia/Nemotron-Mini-4B-Instruct` |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8` |
| `Qwen/Qwen3-4B-Instruct-2507` |

To switch:

```bash
# Update NVIDIA_LLM_MODEL in .env.jetson
nano .env.jetson

# Restart all services (no rebuild needed for model changes)
sudo docker compose -f docker-compose.jetson.yml down
sudo docker compose -f docker-compose.jetson.yml up -d

# Check LLM logs to verify new model is loading
sudo docker compose -f docker-compose.jetson.yml logs -f llm-nvidia-jetson
```

## Common Commands

```bash
# View logs
sudo docker compose -f docker-compose.jetson.yml logs -f python-app

# Stop all services
sudo docker compose -f docker-compose.jetson.yml down

# Rebuild after code changes
sudo docker compose -f docker-compose.jetson.yml up --build -d python-app
```

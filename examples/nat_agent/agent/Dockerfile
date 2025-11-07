# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

# Base image
FROM python:3.12-slim

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="NAT Agent example"
LABEL version="1.0"

# Environment setup
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip uv

# App directory setup
WORKDIR /app

COPY ./agent/ ./agent/

# Example app directory
WORKDIR /app/agent

# Dependencies
RUN uv sync --frozen

# Port configuration
EXPOSE 8005

# Start command
CMD ["uv", "run", "nat", "serve", "--config_file", "/app/agent/configs/config.yml", "--host", "0.0.0.0"]
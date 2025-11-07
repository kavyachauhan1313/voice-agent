"""Frames for delivering system prompt over RTVI data channel.

Provides a frame that carries the default system prompt so downstream
processors (e.g., RTVI transcript/output serializers) can send it to
the UI over the data channel instead of using an HTTP API.
"""

from pipecat.frames.frames import Frame


class ShowSystemPromptFrame(Frame):
    """Frame that carries the default system prompt to be shown in the UI."""

    def __init__(self, prompt: str):
        """Initialize the frame with the system prompt.

        Args:
            prompt: The full system prompt text.
        """
        super().__init__()
        self.prompt = prompt

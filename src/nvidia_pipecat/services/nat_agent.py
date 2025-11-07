# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License
"""NVIDIA Nemo NATAgent service implementation.

Integrates with NVIDIA's Nemo NATAgent service to enhance responses
by incorporating knowledge from external documents. Features include:
    - OpenAI-compatible LLM interface
    - Configurable retrieval parameters
"""

import json
import random
import re
from pathlib import Path

import httpx
import yaml
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSSpeakFrame,
    UserImageRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from rapidfuzz.distance import Levenshtein


class NATAgentService(OpenAILLMService):
    """This is the base class for all services that use NVIDIA Nemo NATAgent.

    Requires deployed NVIDIA NATAgent server. For deployment instructions see:
    Attributes:
        collection_name: Document collection identifier.
        agent_server_url: NATAgent API endpoint URL.
        temperature: Controls response randomness (0-1).
        top_p: Token probability threshold (0-1).
        max_tokens: Maximum response length.
        use_knowledge_base: Whether to use NATAgent retrieval.
        config_file: Config File Path containing tool information for NATAgent.
    """

    _shared_session: httpx.AsyncClient | None = None

    def __init__(
        self,
        agent_server_url: str = "http://localhost:8081",
        stop_words: list | None = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1000,
        use_knowledge_base: bool = True,
        config_file: str | None = None,
        **kwargs,
    ):
        """Initialize the NVIDIA NATAgent service.

        Args:
            collection_name: Document collection identifier.
            agent_server_url: NATAgent API endpoint URL.
            stop_words: Words that stop LLM generation.
            temperature: Controls response randomness (0-1).
            top_p: Token probability threshold (0-1).
            max_tokens: Maximum response length.
            use_knowledge_base: Whether to use NATAgent retrieval.
            config_file: Config File Path containing tool information for NATAgent.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        super().__init__(api_key="", **kwargs)
        self.agent_server_url = agent_server_url
        if stop_words is None:
            stop_words = []
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.use_knowledge_base = use_knowledge_base

        self._current_task = None
        self.config_file = config_file
        self._tools = self._load_tools_from_config(config_file)
        # Track last used phrase to avoid consecutive repetition
        self._last_used_phrase = None
        # Track last query for workaround with different interims and finals from ASR
        self._last_query = None
        # Default natural phrases for tools without specific phrases
        self._default_phrases = [
            "Let me help you with that.",
            "I'll take care of that for you.",
            "Just a moment, working on it.",
            "Sure thing, let me handle that.",
            "On it! Give me a second.",
            "I'm processing that now.",
            "Let me get that sorted for you.",
            "Working on your request.",
            "I'll get that done right away.",
            "Let me take care of this.",
        ]
        self.current_action = None
        self.current_thought = None
        self.last_index = 0

    def _load_tools_from_config(self, config_file: str | None) -> dict:
        """Load tools and their natural phrases from config file.

        Args:
            config_file: Path to the YAML config file containing tool definitions

        Returns:
            Dictionary with tool names as keys and tool config as values
            Format: {"tool_name": {"name": "Tool Name", "natural_phrases": [...]}}
        """
        if config_file is None:
            logger.warning("No config file provided, using empty tools dictionary")
            return {}

        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_file}")
                return {}

            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Extract functions section which contains tool definitions
            functions = config.get("functions", {})
            tools_dict = {}

            for tool_name, tool_config in functions.items():
                natural_phrases = tool_config.get("natural_phrases", [])
                # Handle case where natural_phrases is None (empty YAML value)
                if natural_phrases is None:
                    natural_phrases = []
                tools_dict[tool_name] = {"name": tool_name, "natural_phrases": natural_phrases}

            logger.info(f"Loaded {len(tools_dict)} tools from config file: {config_file}")
            return tools_dict

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {config_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return {}

    @property
    def shared_session(self) -> httpx.AsyncClient:
        """Get the shared HTTP client session.

        Returns:
            httpx.AsyncClient: The shared session for making HTTP requests.
            Creates a new session if none exists.
        """
        if NATAgentService._shared_session is None:
            NATAgentService._shared_session = httpx.AsyncClient()
        return NATAgentService._shared_session

    @shared_session.setter
    def shared_session(self, shared_session: httpx.AsyncClient):
        """Set the shared HTTP client session.

        Args:
            shared_session: The httpx.AsyncClient to use for all instances.
        """
        NATAgentService._shared_session = shared_session

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA NATAgent service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._current_task:
            await self.cancel_task(self._current_task)
        self.current_action = None
        self.current_thought = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA NATAgent service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        if self._current_task:
            await self.cancel_task(self._current_task)

    async def cleanup(self):
        """Clean up resources used by the NATAgent service.

        Closes the shared HTTP client session if it exists and performs parent cleanup.
        """
        await super().cleanup()
        await self._close_client_session()

    async def _close_client_session(self):
        """Close the Client Session if it exists."""
        if NATAgentService._shared_session:
            await NATAgentService._shared_session.aclose()
            NATAgentService._shared_session = None

    async def test_connection(self) -> bool:
        """Test if the NATAgent service is reachable.

        Returns:
            bool: True if service is reachable, False otherwise
        """
        try:
            logger.info(f"Testing connection to NATAgent at {self.agent_server_url}")
            # Try a simple GET request to see if the service is up
            resp = await self.shared_session.get(
                f"{self.agent_server_url}/health",  # Common health endpoint
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info("NATAgent service is reachable")
                return True
            else:
                logger.warning(f"NATAgent health check returned {resp.status_code}")
                return False
        except httpx.ConnectError:
            logger.error(f"Cannot connect to NATAgent at {self.agent_server_url}")
            return False
        except httpx.TimeoutException:
            logger.error(f"Timeout connecting to NATAgent at {self.agent_server_url}")
            return False
        except Exception as e:
            logger.error(f"Error testing NATAgent connection: {type(e).__name__}: {e}")
            return False

    async def _get_natagent_response(self, request_json: dict):
        """Make HTTP request to NATAgent endpoint with proper error handling."""
        try:
            logger.debug(f"NAT Agent: Sending POST request to {self.agent_server_url}/chat/stream")
            resp = await self.shared_session.post(
                f"{self.agent_server_url}/chat/stream",
                json=request_json,
                timeout=30.0,  # Add explicit timeout
            )
            # Check if response was successful
            resp.raise_for_status()
            logger.debug(f"NAT Agent: Successfully received response: {resp.status_code}")
            return resp
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from NATAgent: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to NATAgent at {self.agent_server_url}: {e}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to NATAgent: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in NATAgent request: {type(e).__name__}: {e}")
            raise

    def _sanitize_text_for_tts(self, text: str) -> str:
        """Sanitize text to prevent TTS errors with special characters.

        Args:
            text: Raw text from LLM response
        Returns:
            Sanitized text safe for TTS processing
        """
        if not text:
            return ""
        # Remove or replace problematic characters that can cause TTS issues
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        # Replace multiple consecutive whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Remove or replace common problematic patterns
        # Fix unmatched brackets/quotes that might cause parsing issues
        brackets = {"[": "]", "{": "}", "(": ")"}
        quotes = ['"', "'", "`"]
        # Simple bracket matching - remove unmatched opening brackets at end
        for open_char, close_char in brackets.items():
            if text.count(open_char) > text.count(close_char):
                # Remove trailing unmatched opening brackets
                while text.endswith(open_char):
                    text = text[:-1].strip()
        # Remove unmatched quotes at the end
        for quote in quotes:
            if text.count(quote) % 2 == 1 and text.endswith(quote):
                text = text[:-1].strip()
        # Remove markdown-style formatting that might confuse TTS
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # *italic*
        text = re.sub(r"`(.*?)`", r"\1", text)  # `code`
        # Handle Python dictionary/list formats that cause TTS issues
        # Convert dict/list representations to more natural speech
        text = self._format_data_structures_for_speech(text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Clean up extra spaces
        text = text.strip()
        return text

    def _format_data_structures_for_speech(self, text: str) -> str:
        """Convert Python data structures to speech-friendly format."""
        # Pattern to match dictionary-like structures
        dict_pattern = r"\{[^{}]*\}"
        list_pattern = r"\[[^\[\]]*\]"

        def format_dict_match(match):
            dict_str = match.group(0)
            try:
                # Try to parse as a dictionary-like structure
                # Replace single quotes with double quotes for JSON parsing
                dict_str_json = dict_str.replace("'", '"')
                parsed = json.loads(dict_str_json)
                if isinstance(parsed, dict):
                    # Convert to natural speech format
                    items = []
                    for key, value in parsed.items():
                        if isinstance(value, int | float):
                            items.append(f"{key} costs {value} dollars")
                        else:
                            items.append(f"{key} is {value}")
                    return ", ".join(items)
                elif isinstance(parsed, list):
                    return ", ".join(str(item) for item in parsed)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, just clean up the formatting
                cleaned = dict_str.replace("{", "").replace("}", "")
                cleaned = cleaned.replace("[", "").replace("]", "")
                cleaned = cleaned.replace("'", "")
                return cleaned
            return dict_str

        def format_list_match(match):
            list_str = match.group(0)
            try:
                # Clean up list formatting
                cleaned = list_str.replace("[", "").replace("]", "")
                cleaned = cleaned.replace("'", "")
                return cleaned
            except Exception:
                return list_str

        # Apply transformations
        text = re.sub(dict_pattern, format_dict_match, text)
        text = re.sub(list_pattern, format_list_match, text)
        return text

    def _extract_intermediate_data(self, payload: dict) -> tuple[str, str]:
        """Extract output and thought from intermediate data payload.

        Args:
            payload: Dictionary containing intermediate data
        Returns:
            Tuple of (output_text, thought_text)
        """
        output = ""
        thought = ""
        # Extract text after **Output**
        if isinstance(payload, str):
            output_match = re.search(r"\*\*Output\*\*(.*?)(?=\*\*|$)", payload, re.DOTALL)
            if output_match:
                output = output_match.group(1).strip()
            # Extract Thought content
            thought_match = re.search(r"Thought:(.*?)(?=\*\*|$)", payload, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()
        return output, thought

    def get_random_message(self, natural_phrases):
        """Get a random natural phrase without consecutive repetition.

        Args:
            natural_phrases: List of natural phrases to choose from
        Returns:
            A natural phrase, or a default message if no phrases available
        """
        # Use default phrases if the provided list is empty
        if not natural_phrases:
            natural_phrases = self._default_phrases
        # If there's only one phrase, return it
        if len(natural_phrases) == 1:
            selected_phrase = natural_phrases[0]
        else:
            # Filter out the last used phrase to avoid consecutive repetition
            available_phrases = [phrase for phrase in natural_phrases if phrase != self._last_used_phrase]

            # If filtering leaves us with no options (shouldn't happen with >1 phrase), use all
            if not available_phrases:
                available_phrases = natural_phrases
            # Randomly select from available phrases
            selected_phrase = random.choice(available_phrases)
        # Update the last used phrase
        self._last_used_phrase = selected_phrase
        return selected_phrase

    async def _process_context(self, context: OpenAILLMContext):
        """Processes LLM context through NATAgent pipeline.

        Args:
            context: Contains conversation history and settings.

        Raises:
            Exception: If invalid message role or empty query.
        """
        try:
            messages: list[ChatCompletionMessageParam] = context.get_messages()

            # Separate conversation history (all except last) and current query
            if not messages:
                raise Exception("No query is provided..")

            conversation_parts = []
            if len(messages) > 1:
                # Process all messages except the last one as conversation history
                for msg in messages[:-1]:
                    if msg["role"] != "system" and msg["role"] != "user" and msg["role"] != "assistant":
                        raise Exception(f"Unexpected role {msg['role']} found!")

                    role = msg["role"]
                    content = msg.get("content", "").strip()
                    if content:  # Only include messages with content
                        conversation_parts.append(f'{role} said "{content}"')

            # Get the last message as current query
            last_message = messages[-1]
            if (
                last_message["role"] != "system"
                and last_message["role"] != "user"
                and last_message["role"] != "assistant"
            ):
                raise Exception(f"Unexpected role {last_message['role']} found!")

            current_query = last_message.get("content", "").strip()
            if not current_query:
                raise Exception("Current query is empty..")

            # Combine conversation history and current query
            conversation_content_parts = []
            if conversation_parts:
                conversation_content_parts.append(", ".join(conversation_parts))
            conversation_content_parts.append(f"Current Query: {current_query}")

            conversation_content = ". ".join(conversation_content_parts)

            logger.debug(f"NAT Agent: Converted conversation content: {conversation_content}")

            if not conversation_content:
                raise Exception("No query is provided..")

            # Send as single user message with conversation history (like frontend)
            chat_details = [{"role": "user", "content": conversation_content}]
            """
            Call the NATAgent chain server and return the streaming response.
            """
            request_json = {
                "messages": chat_details,
                "use_knowledge_base": self.use_knowledge_base,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            # Validate configuration before making request
            if not self.agent_server_url:
                raise ValueError("NATAgent server URL is not configured")
            if not hasattr(self, "shared_session") or self.shared_session is None:
                raise RuntimeError("HTTP session is not initialized")
            # Debug logging for the request
            logger.debug(f"NAT Agent: Making request to NATAgent endpoint: {self.agent_server_url}/chat/stream")
            logger.debug(f"NAT Agent: Request payload keys: {list(request_json.keys())}")
            logger.debug(f"NAT Agent: Messages count: {len(request_json.get('messages', []))}")
            logger.debug(f"NAT Agent: Use knowledge base: {request_json.get('use_knowledge_base', False)}")
            filler_sent = False
            full_response = ""
            await self.start_ttfb_metrics()
            resp = await self._get_natagent_response(request_json)
            logger.debug(f"NAT Agent: Received response with status: {resp.status_code}")
            logger.debug(f"NAT Agent: Response headers: {dict(resp.headers)}")
            try:
                async for chunk in resp.aiter_lines():
                    try:
                        chunk = chunk.strip()
                        if not chunk:
                            logger.debug("NAT Agent: Empty chunk, skipping")
                            continue
                        await self.stop_ttfb_metrics()
                        # Handle different chunk prefixes
                        message = ""
                        if chunk.startswith("data: "):
                            # Handle regular data chunks
                            chunk = chunk[6:].strip()  # Remove "data: " prefix
                            try:
                                parsed = json.loads(chunk)
                                choices = parsed.get("choices", [])
                                if choices:
                                    message_data = choices[0].get("message", {})
                                    message = message_data.get("content", "")
                            except json.JSONDecodeError as e:
                                logger.error(f"NAT Agent: Failed to parse data chunk: {e}")
                                continue
                        elif chunk.startswith("intermediate_data: "):
                            logger.debug("We got intermediate data")
                            # Handle intermediate data chunks
                            chunk = chunk[19:].strip()  # Remove "intermediate_data: " prefix
                            try:
                                parsed = json.loads(chunk)
                                payload = parsed.get("payload", "")
                                # First find the **Output** section if it exists
                                output_section_match = re.search(r"\*\*Output:\*\*(.*?)(?=\*\*|$)", payload, re.DOTALL)
                                if output_section_match:
                                    output_text = output_section_match.group(1).strip()
                                    # Look for single-line Thought and Action
                                    thought_match = re.search(r"Thought:(.*?)(?=\n|$)", output_text)
                                    action_match = re.search(r"Action:(.*?)(?=\n|$)", output_text)
                                    if thought_match:
                                        thought = thought_match.group(1).strip()
                                    if action_match:
                                        action = action_match.group(1).strip()
                                        if action in self._tools and (
                                            (self.current_action is None or self.current_thought is None)
                                            or (self.current_action != action and self.current_thought != thought)
                                        ):
                                            self.current_action = action
                                            self.current_thought = thought
                                            random_message = self.get_random_message(
                                                self._tools[action]["natural_phrases"]
                                            )
                                            if not filler_sent:
                                                logger.debug(
                                                    f"NAT Agent: Sending Filler message {random_message} "
                                                    f"on Action {action} downstream"
                                                )
                                                await self.push_frame(TTSSpeakFrame(random_message))
                                                filler_sent = True
                                            # # Add full stop if not present
                                            # thought_with_stop = (
                                            #     thought if thought.endswith('.') else f"{thought}."
                                            # )
                                            # await self.push_frame(TextFrame(f"{thought_with_stop} "))
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse intermediate chunk: {e}")
                                continue
                        else:
                            logger.warning(f"Unknown chunk format: {repr(chunk)}")
                            continue
                        if not message:
                            logger.debug("No content extracted from chunk")
                            continue
                        # Sanitize the message for TTS
                        sanitized_message = self._sanitize_text_for_tts(message)
                        logger.debug(f"NAT Agent: Original message - {repr(message)}")
                        logger.debug(f"NAT Agent: Sanitized message - {repr(sanitized_message)}")
                    except Exception as e:
                        logger.error(f"Parsing NATAgent response chunk failed. Error: {e}")
                        continue
                    full_response += sanitized_message
                    if sanitized_message:
                        await self.push_frame(TextFrame(sanitized_message))
            except Exception as e:
                await self.push_error(ErrorFrame("Internal error in NATAgent stream: " + str(e)))
            finally:
                await resp.aclose()
            logger.debug(f"NAT Agent: Full NATAgent response - {full_response}")
        except Exception as e:
            logger.error(f"An error occurred in http request to NATAgent endpoint, Error:  {e}")
            await self.push_error(
                ErrorFrame("An error occurred in http request to NATAgent endpoint, Error: " + str(e))
            )

    async def _update_settings(self, settings):
        """Updates service settings.

        Args:
            settings: Dictionary of setting name-value pairs.
        """
        for setting, value in settings.items():
            logger.debug(f"NAT Agent: Updating {setting} to {value} via NATAgentSettingsFrame")
            match setting:
                case "agent_server_url":
                    self.agent_server_url = value
                case "temperature":
                    self.temperature = value
                case "top_p":
                    self.top_p = value
                case "max_tokens":
                    self.max_tokens = value
                case "use_knowledge_base":
                    self.use_knowledge_base = value
                case _:
                    logger.warning(f"Unknown setting for NATAgent service: {setting}")

    async def _process_context_and_frames(self, context: OpenAILLMContext):
        """Process context and handle start/end frames with metrics."""
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self._process_context(context)
        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes pipeline frames.

        Handles settings updates and parent frame processing.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        context = None
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._last_query = None
        # if isinstance(frame, NATAgentSettingsFrame):
        #     await self._update_settings(frame.settings)
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, UserImageRawFrame):
            context = OpenAILLMContext()
            context.add_image_frame_message(format=frame.format, size=frame.size, image=frame.image, text=frame.text)
        elif isinstance(frame, StartInterruptionFrame):
            logger.debug("NATAgent: Got a startinterruption frame.")
            if self._current_task is not None:
                await self.cancel_task(self._current_task)
            await self._start_interruption()
            await self.stop_all_metrics()
            await self.push_frame(frame)
        else:
            await super().process_frame(frame, direction)

        if context:
            # Check for similarity with last query BEFORE starting new task
            messages = context.get_messages()
            if messages and len(messages) > 0:
                current_query = messages[-1]["content"].strip()
                logger.debug(f"NAT Agent: Current query is '{current_query}'")
                logger.debug(f"NAT Agent: The Last Query is '{self._last_query}'.")

                if self._last_query is not None:
                    sim = 1 - Levenshtein.normalized_distance(current_query, self._last_query)
                    logger.debug(
                        f"NAT Agent: The similarity between '{current_query}' and '{self._last_query}' is {sim}"
                    )
                    if sim >= 0.90:
                        logger.debug("NAT Agent: Not proceeding with the query as its similar to last one")
                        # Don't start new task for similar query
                        return

                # Update last query for future comparisons
                self._last_query = current_query

            # Cancel current task before starting new one
            new_task = self.create_task(self._process_context_and_frames(context))
            if self._current_task is not None:
                await self.cancel_task(self._current_task)
            self._current_task = new_task
            self._current_task.add_done_callback(lambda _: setattr(self, "_current_task", None))

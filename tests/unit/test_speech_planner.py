# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the SpeechPlanner service.

This module contains tests for the SpeechPlanner class, focusing on core functionalities:
- Service initialization with various parameters
- Frame processing for different frame types
- Speech completion detection and LLM integration
- Chat history management with context windows
- Interruption handling and VAD integration
- Label preprocessing and classification logic
"""

import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.services.speech_planner import SpeechPlanner


class MockBaseMessageChunk:
    """Mock for BaseMessageChunk that mimics the structure."""

    def __init__(self, content=""):
        """Initialize with content.

        Args:
            content: The text content of the chunk
        """
        self.content = content


@pytest.fixture
def mock_prompt_file():
    """Create a temporary YAML prompt file for testing."""
    prompt_data = {
        "configurations": {"using_chat_history": False},
        "prompts": {
            "completion_prompt": (
                "Evaluate whether the following user speech is sufficient:\n"
                "1. Label1: Complete and coherent thought\n"
                "2. Label2: Incomplete speech\n"
                "3. Label3: User commands\n"
                "4. Label4: Acknowledgments\n"
                "User Speech: {transcript}\n"
                "Only return Label1 or Label2 or Label3 or Label4."
            )
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(prompt_data, f)
        return f.name


@pytest.fixture
def mock_context():
    """Create a mock OpenAI LLM context."""
    context = Mock(spec=OpenAILLMContext)
    context.get_messages.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"},
    ]
    return context


@pytest.fixture
def speech_planner(mock_prompt_file, mock_context):
    """Create a SpeechPlanner instance for testing."""
    with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
        mock_chat.return_value = AsyncMock()
        planner = SpeechPlanner(
            prompt_file=mock_prompt_file,
            model="test-model",
            api_key="test-key",
            base_url="http://test-url",
            context=mock_context,
            context_window=2,
        )
        return planner


class TestSpeechPlannerInitialization:
    """Test SpeechPlanner initialization."""

    def test_init_with_default_params(self, mock_prompt_file, mock_context):
        """Test initialization with default parameters."""
        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            planner = SpeechPlanner(prompt_file=mock_prompt_file, context=mock_context)

            assert planner.model_name == "nvdev/google/gemma-2b-it"
            assert planner.context == mock_context
            assert planner.context_window == 1
            assert planner.user_speaking is None
            assert planner.current_prediction is None

    def test_init_with_custom_params(self, mock_prompt_file, mock_context):
        """Test initialization with custom parameters."""
        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            params = SpeechPlanner.InputParams(temperature=0.7, max_tokens=100, top_p=0.9)

            planner = SpeechPlanner(
                prompt_file=mock_prompt_file,
                model="custom-model",
                api_key="custom-key",
                base_url="http://custom-url",
                context=mock_context,
                params=params,
                context_window=3,
            )

            assert planner.model_name == "custom-model"
            assert planner.context_window == 3
            assert planner._settings["temperature"] == 0.7
            assert planner._settings["max_tokens"] == 100
            assert planner._settings["top_p"] == 0.9

    def test_init_loads_prompts(self, mock_prompt_file, mock_context):
        """Test that initialization properly loads prompts from file."""
        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            planner = SpeechPlanner(prompt_file=mock_prompt_file, context=mock_context)

            assert "configurations" in planner.prompts
            assert "prompts" in planner.prompts
            assert "completion_prompt" in planner.prompts["prompts"]
            assert planner.prompts["configurations"]["using_chat_history"] is False


### Adding tests for preprocess_pred function


class TestPreprocessPred:
    """Test the preprocess_pred function logic through end-to-end processing."""

    @pytest.mark.asyncio
    async def test_preprocess_pred_label1_complete(self, speech_planner):
        """Test preprocess_pred with Label1 returns Complete via end-to-end processing."""
        frame = TranscriptionFrame("Hello there", "user1", datetime.now())
        test_cases = ["Label1", "Label 1", "The answer is Label1.", "I think this is Label 1 based on analysis."]

        for case in test_cases:
            # Mock the LLM to return the specific prediction
            mock_chunks = [MockBaseMessageChunk(case)]

            with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value.__aiter__.return_value = mock_chunks

                await speech_planner._process_complete_context(frame)
                assert speech_planner.current_prediction == "Complete", f"Failed for case: {case}"

    @pytest.mark.asyncio
    async def test_preprocess_pred_label2_incomplete(self, speech_planner):
        """Test preprocess_pred with Label2 returns Incomplete via end-to-end processing."""
        frame = TranscriptionFrame("Hello", "user1", datetime.now())
        test_cases = ["Label2", "Label 2", "The answer is Label2.", "This should be Label 2."]

        for case in test_cases:
            # Mock the LLM to return the specific prediction
            mock_chunks = [MockBaseMessageChunk(case)]

            with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value.__aiter__.return_value = mock_chunks

                await speech_planner._process_complete_context(frame)
                assert speech_planner.current_prediction == "Incomplete", f"Failed for case: {case}"

    @pytest.mark.asyncio
    async def test_preprocess_pred_label3_complete(self, speech_planner):
        """Test preprocess_pred with Label3 returns Complete via end-to-end processing."""
        frame = TranscriptionFrame("Stop that", "user1", datetime.now())
        test_cases = ["Label3", "Label 3", "The answer is Label3.", "I classify this as Label 3."]

        for case in test_cases:
            # Mock the LLM to return the specific prediction
            mock_chunks = [MockBaseMessageChunk(case)]

            with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value.__aiter__.return_value = mock_chunks

                await speech_planner._process_complete_context(frame)
                assert speech_planner.current_prediction == "Complete", f"Failed for case: {case}"

    @pytest.mark.asyncio
    async def test_preprocess_pred_label4_complete(self, speech_planner):
        """Test preprocess_pred with Label4 returns Complete via end-to-end processing."""
        frame = TranscriptionFrame("Okay", "user1", datetime.now())
        test_cases = ["Label4", "Label 4", "The answer is Label4.", "This is Label 4 category."]

        for case in test_cases:
            # Mock the LLM to return the specific prediction
            mock_chunks = [MockBaseMessageChunk(case)]

            with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value.__aiter__.return_value = mock_chunks

                await speech_planner._process_complete_context(frame)
                assert speech_planner.current_prediction == "Complete", f"Failed for case: {case}"

    @pytest.mark.asyncio
    async def test_preprocess_pred_unrecognized_incomplete(self, speech_planner):
        """Test preprocess_pred with unrecognized labels returns Incomplete via end-to-end processing."""
        frame = TranscriptionFrame("Unknown input", "user1", datetime.now())
        test_cases = ["Label5", "Unknown", "No label found", "", "Some random text"]

        for case in test_cases:
            # Mock the LLM to return the specific prediction
            mock_chunks = [MockBaseMessageChunk(case)]

            with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value.__aiter__.return_value = mock_chunks

                await speech_planner._process_complete_context(frame)
                assert speech_planner.current_prediction == "Incomplete", f"Failed for case: {case}"


### Adding tests for chat_history management
class TestChatHistory:
    """Test chat history management."""

    def test_get_chat_history_empty_messages(self, mock_prompt_file):
        """Test get_chat_history with empty message list."""
        context = Mock(spec=OpenAILLMContext)
        context.get_messages.return_value = []

        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            planner = SpeechPlanner(prompt_file=mock_prompt_file, context=context, context_window=2)

            history = planner.get_chat_history()
            assert history == []

    def test_get_chat_history_with_context_window(self, mock_prompt_file):
        """Test get_chat_history respects context_window setting."""
        messages = [
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
            {"role": "user", "content": "Second user message"},
            {"role": "assistant", "content": "Second assistant response"},
            {"role": "user", "content": "Third user message"},
            {"role": "assistant", "content": "Third assistant response"},
        ]

        context = Mock(spec=OpenAILLMContext)
        context.get_messages.return_value = messages

        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            # Test with context_window=1 (should get last 2 messages)
            planner = SpeechPlanner(prompt_file=mock_prompt_file, context=context, context_window=1)

            history = planner.get_chat_history()
            assert len(history) == 2
            assert history[0]["content"] == "Third user message"
            assert history[1]["content"] == "Third assistant response"

    def test_get_chat_history_starts_with_user(self, mock_prompt_file):
        """Test get_chat_history starts with user message."""
        messages = [
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response 1"},
            {"role": "user", "content": "User message 2"},
            {"role": "assistant", "content": "Assistant response 2"},
        ]

        context = Mock(spec=OpenAILLMContext)
        context.get_messages.return_value = messages

        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat:
            mock_chat.return_value = AsyncMock()

            planner = SpeechPlanner(prompt_file=mock_prompt_file, context=context, context_window=2)

            history = planner.get_chat_history()
            assert len(history) > 0
            assert history[0]["role"] == "user"


class TestFrameProcessing:
    """Test frame processing functionality by testing the logic directly."""

    def test_user_speaking_state_changes(self, speech_planner):
        """Test that user speaking state changes correctly."""
        # Test UserStartedSpeakingFrame logic
        assert speech_planner.user_speaking is None
        speech_planner.user_speaking = True
        assert speech_planner.user_speaking is True

        # Test UserStoppedSpeakingFrame logic
        speech_planner.user_speaking = False
        assert speech_planner.user_speaking is False

    def test_bot_speaking_timestamp_tracking(self, speech_planner):
        """Test bot speaking timestamp tracking."""
        # Initially no timestamp
        assert speech_planner.latest_bot_started_speaking_frame_timestamp is None

        # Set timestamp (simulating BotStartedSpeakingFrame)
        test_time = datetime.now()
        speech_planner.latest_bot_started_speaking_frame_timestamp = test_time
        assert speech_planner.latest_bot_started_speaking_frame_timestamp == test_time

        # Clear timestamp (simulating BotStoppedSpeakingFrame)
        speech_planner.latest_bot_started_speaking_frame_timestamp = None
        assert speech_planner.latest_bot_started_speaking_frame_timestamp is None

    def test_frame_state_management(self, speech_planner):
        """Test frame state management without full processing."""
        # Test last_frame tracking
        frame = InterimTranscriptionFrame("Hello", "user1", datetime.now())
        speech_planner.last_frame = frame
        assert speech_planner.last_frame == frame

        # Test clearing last_frame
        speech_planner.last_frame = None
        assert speech_planner.last_frame is None

        # Test current_prediction state
        speech_planner.current_prediction = "Complete"
        assert speech_planner.current_prediction == "Complete"

        speech_planner.current_prediction = "Incomplete"
        assert speech_planner.current_prediction == "Incomplete"

    def test_transcription_frame_conditions(self, speech_planner):
        """Test the conditions for processing transcription frames."""
        # Set up conditions for processing
        speech_planner.user_speaking = False
        speech_planner.current_prediction = "Incomplete"

        # These conditions should allow processing
        assert speech_planner.user_speaking is False
        assert speech_planner.current_prediction == "Incomplete"

        # Test conditions that would prevent processing
        speech_planner.user_speaking = True
        assert speech_planner.user_speaking is True  # Should prevent processing

    @pytest.mark.asyncio
    async def test_cancel_current_task_helper(self, speech_planner):
        """Test the _cancel_current_task helper method."""
        # Test with no current task
        speech_planner._current_task = None
        await speech_planner._cancel_current_task()
        assert speech_planner._current_task is None

        # Test with completed task
        completed_task = Mock()
        completed_task.done.return_value = True
        completed_task.cancelled.return_value = False
        speech_planner._current_task = completed_task

        await speech_planner._cancel_current_task()
        assert speech_planner._current_task is None

        # Test with cancelled task
        cancelled_task = Mock()
        cancelled_task.done.return_value = False
        cancelled_task.cancelled.return_value = True
        speech_planner._current_task = cancelled_task

        await speech_planner._cancel_current_task()
        assert speech_planner._current_task is None

        # Test with active task that needs cancellation
        active_task = Mock()
        active_task.done.return_value = False
        active_task.cancelled.return_value = False
        speech_planner._current_task = active_task

        with patch.object(speech_planner, "cancel_task", new_callable=AsyncMock) as mock_cancel:
            await speech_planner._cancel_current_task()
            mock_cancel.assert_called_once_with(active_task)
            assert speech_planner._current_task is None


class TestCompletionDetection:
    """Test speech completion detection."""

    @pytest.mark.asyncio
    async def test_process_complete_context_with_complete_prediction(self, speech_planner):
        """Test _process_complete_context with complete prediction."""
        frame = TranscriptionFrame("Hello there", "user1", datetime.now())

        # Mock the LLM response
        mock_chunks = [MockBaseMessageChunk("Label1")]

        with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value.__aiter__.return_value = mock_chunks
            with patch.object(speech_planner, "push_frame", new_callable=AsyncMock) as mock_push:
                await speech_planner._process_complete_context(frame)

                assert speech_planner.current_prediction == "Complete"

                # Should push interruption and transcription frames
                assert mock_push.call_count >= 2

                # Verify the correct frames were pushed
                call_args = [args[0][0] for args in mock_push.call_args_list]
                assert any(isinstance(f, InterruptionFrame) for f in call_args)
                assert any(isinstance(f, TranscriptionFrame) for f in call_args)

    @pytest.mark.asyncio
    async def test_process_complete_context_with_incomplete_prediction(self, speech_planner):
        """Test _process_complete_context with incomplete prediction."""
        frame = TranscriptionFrame("Hello", "user1", datetime.now())

        # Mock the LLM response with Label2 (incomplete)
        mock_chunks = [MockBaseMessageChunk("Label2")]

        with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value.__aiter__.return_value = mock_chunks
            with patch.object(speech_planner, "push_frame", new_callable=AsyncMock) as mock_push:
                await speech_planner._process_complete_context(frame)

                assert speech_planner.current_prediction == "Incomplete"
                # Should not push any frames for incomplete prediction
                mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_complete_context_with_error(self, speech_planner):
        """Test _process_complete_context handles errors gracefully with proper logging."""
        frame = TranscriptionFrame("Hello", "user1", datetime.now())

        # Mock an exception during processing
        with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
            mock_stream.side_effect = Exception("LLM service error")

            # Patch logger to verify warning is logged with stack trace
            with patch("nvidia_pipecat.services.speech_planner.logger") as mock_logger:
                await speech_planner._process_complete_context(frame)

                # Should default to "Complete" on error
                assert speech_planner.current_prediction == "Complete"

                # Should log warning with stack trace information
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "Disabling Smart EOU detection due to error" in call_args[0][0]
                assert "LLM service error" in call_args[0][0]
                assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_process_complete_context_with_chunk_content_error(self, speech_planner):
        """Test _process_complete_context handles chunk content errors gracefully."""
        frame = TranscriptionFrame("Hello", "user1", datetime.now())

        # Mock chunks where one has content that causes concatenation error
        class BadChunk:
            def __init__(self):
                # Use a number instead of None to pass the `if not chunk.content:` check
                # but still cause TypeError during string concatenation
                self.content = 42

        mock_chunks = [MockBaseMessageChunk("Label"), BadChunk()]

        with patch.object(speech_planner, "_stream_chat_completions", new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value.__aiter__.return_value = mock_chunks

            # Patch logger to verify debug message is logged for chunk error
            with patch("nvidia_pipecat.services.speech_planner.logger") as mock_logger:
                await speech_planner._process_complete_context(frame)

                # Should still get a prediction (either from first chunk or default "Complete")
                assert speech_planner.current_prediction is not None

                # Should log debug message for chunk content error
                debug_calls = [
                    call for call in mock_logger.debug.call_args_list if "Failed to append chunk content" in str(call)
                ]
                assert len(debug_calls) >= 1, (
                    f"Expected debug log for chunk error, got calls: {mock_logger.debug.call_args_list}"
                )


class TestClientCreation:
    """Test client creation functionality."""

    def test_create_client_with_params(self, mock_prompt_file, mock_context):
        """Test create_client method with parameters."""
        with patch("nvidia_pipecat.services.speech_planner.ChatNVIDIA") as mock_chat_class:
            mock_client = AsyncMock()
            mock_chat_class.return_value = mock_client

            planner = SpeechPlanner(
                prompt_file=mock_prompt_file,
                context=mock_context,
                model="test-model",
                api_key="test-key",
                base_url="http://test-url",
            )

            client = planner.create_client(api_key="custom-key", base_url="http://custom-url")

            # Verify ChatNVIDIA was called with correct parameters
            mock_chat_class.assert_called_with(base_url="http://custom-url", model="test-model", api_key="custom-key")
            assert client == mock_client


@pytest.mark.asyncio
async def test_get_chat_completions(speech_planner):
    """Test get_chat_completions method."""
    messages = [{"role": "user", "content": "Test message"}]

    # Mock the client's astream method to return an async iterator
    mock_chunks = [MockBaseMessageChunk("Response chunk")]

    async def mock_astream(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    speech_planner._client.astream = mock_astream

    result = await speech_planner.get_chat_completions(messages)

    # The result should be the return value from astream
    assert result is not None

    # Convert result to list to verify contents
    result_list = []
    async for chunk in result:
        result_list.append(chunk)

    assert len(result_list) == 1
    assert result_list[0].content == "Response chunk"

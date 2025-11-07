# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for action frame creation and manipulation.

This module tests the creation, validation, and comparison of various action frames used for bot and user actions.
"""

# ruff: noqa: F405

import pytest
from pipecat.frames.frames import TextFrame

from nvidia_pipecat.frames.action import (
    FinishedActionFrame,
    StartActionFrame,
    StartedActionFrame,
    StartedPresenceUserActionFrame,
    StopActionFrame,
)
from tests.unit.utils import ignore_ids


def test_action_frame_basic_usage():
    """Tests basic action frame functionality.

    Tests:
        - Frame creation with parameters
        - Action ID propagation
        - Frame name generation
        - Frame attribute access

    Raises:
        AssertionError: If frame attributes don't match expected values.
    """
    start_frame = StartActionFrame()
    action_id = start_frame.action_id

    started_frame = StartedActionFrame(action_id=action_id)
    stop_frame = StopActionFrame(action_id=action_id)
    finished_frame = FinishedActionFrame(action_id=action_id)

    assert started_frame.action_id == action_id
    assert stop_frame.action_id == action_id
    assert stop_frame.name == "StopActionFrame#0"
    assert finished_frame.action_id == action_id
    assert finished_frame.name == "FinishedActionFrame#0"


def test_required_parameters():
    """Tests parameter validation in frame creation.

    Tests:
        - Required parameter enforcement
        - Type checking
        - Error handling

    Raises:
        TypeError: When required parameters are missing.
    """
    with pytest.raises(TypeError):
        StartedPresenceUserActionFrame()  # type: ignore


def test_frame_comparison_ignoring_ids():
    """Tests frame comparison with ID ignoring.

    Tests:
        - Frame equality comparison
        - ID-independent comparison
        - Content-based comparison

    Raises:
        AssertionError: If frame comparison results are incorrect.
    """
    a = TextFrame(text="test")
    b = TextFrame(text="test")
    c = TextFrame(text="something")

    assert a != b
    assert a == ignore_ids(b)
    assert a != ignore_ids(c)
